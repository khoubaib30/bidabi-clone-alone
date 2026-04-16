"""
Training pipeline for ResNet-18 fine-tuning on a pre-split dataset.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not installed - skipping UMAP visualization.")
    UMAP_AVAILABLE = False


H = 256
W = 256
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 3
SEED = 42

DATA_DIR = Path("data/processed")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model_resnet18_finetuned.pth"
PLOTS_DIR = Path("reports/figures")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def is_valid_image(path: str) -> bool:
    return path.lower().endswith(VALID_EXTENSIONS)


def create_dataset(root: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return datasets.ImageFolder(
        root=root,
        transform=transform,
        is_valid_file=is_valid_image,
    )


def create_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    preds = []
    labels = []
    probs = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            p = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return np.array(preds), np.array(labels), np.array(probs)


def compute_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                correct[label] += int(predicted[i].item() == label)
                total[label] += 1

    return np.divide(correct, total, out=np.zeros_like(correct), where=total != 0)


def plot_confusion_matrix(cm: np.ndarray, classes: list[str]) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (ResNet18 full FT + MixUp)")
    save_plot("confusion_matrix.png")


def plot_roc_curves(labels: np.ndarray, probs: np.ndarray, classes: list[str]) -> None:
    y_bin = label_binarize(labels, classes=list(range(len(classes))))

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (ResNet18 full FT + MixUp)")
    plt.legend()
    save_plot("roc_curves.png")


def compute_hardest_samples(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
    top_k: int = 12,
) -> None:
    criterion_nr = nn.CrossEntropyLoss(reduction="none")
    losses = []
    imgs = []
    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            batch_losses = criterion_nr(outputs, y)

            losses.extend(batch_losses.cpu().numpy())
            imgs.extend(images.cpu())
            labels.extend(y.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())

    losses = np.array(losses)
    idx_sorted = np.argsort(losses)[::-1]
    top_k = min(top_k, len(idx_sorted))

    plt.figure(figsize=(12, 10))
    for i in range(top_k):
        idx = idx_sorted[i]
        img = imgs[idx].permute(1, 2, 0).numpy()
        img = (
            img * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(
            f"Loss={losses[idx]:.2f}\n"
            f"True={classes[labels[idx]]}\n"
            f"Pred={classes[preds[idx]]}"
        )
        plt.axis("off")

    plt.suptitle("Top Hardest Samples (Highest Loss)")
    save_plot("hardest_samples.png")


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    return torch.flatten(x, 1)


def save_plot(filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Expected train/val/test directories under data/processed. "
            "Run src/prepare_dataset.py first."
        )

    train_dataset = create_dataset(TRAIN_DIR, train_transform)
    val_dataset = create_dataset(VAL_DIR, test_transform)
    test_dataset = create_dataset(TEST_DIR, test_transform)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Train/val/test splits do not expose the same classes.")

    dataset_classes = train_dataset.classes
    num_classes = len(dataset_classes)
    print("Detected classes:", dataset_classes)
    print(
        f"Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = create_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (
                lam * criterion(outputs, targets_a)
                + (1 - lam) * criterion(outputs, targets_b)
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved to {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print("Training completed. Best validation accuracy:", best_val_acc)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (ResNet18 full FT + MixUp)")
    plt.legend()
    save_plot("training_validation_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (ResNet18 full FT + MixUp)")
    plt.legend()
    save_plot("validation_accuracy.png")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device)

    print(classification_report(all_labels, all_preds, target_names=dataset_classes))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, dataset_classes)

    per_class_acc = compute_per_class_accuracy(
        model,
        test_loader,
        num_classes,
        device,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(dataset_classes, per_class_acc)
    plt.ylabel("Accuracy")
    plt.title("Per-class Accuracy (ResNet18 full FT + MixUp)")
    plt.xticks(rotation=45)
    save_plot("per_class_accuracy.png")

    plot_roc_curves(all_labels, all_probs, dataset_classes)
    compute_hardest_samples(model, test_loader, dataset_classes, device)

    embeddings = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = extract_features(model, images)
            embeddings.append(feats.cpu().numpy())
            labels_list.extend(labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels_list = np.array(labels_list)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_emb = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(dataset_classes):
        idx = labels_list == i
        plt.scatter(tsne_emb[idx, 0], tsne_emb[idx, 1], label=cls, alpha=0.6)

    plt.legend()
    plt.title("t-SNE Embedding Visualization")
    save_plot("tsne_embedding.png")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
        umap_emb = reducer.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(dataset_classes):
            idx = labels_list == i
            plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1], label=cls, alpha=0.6)

        plt.legend()
        plt.title("UMAP Embedding Visualization")
        save_plot("umap_embedding.png")


if __name__ == "__main__":
    main()
