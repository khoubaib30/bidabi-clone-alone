# BIDABI Clone Adapt Create - Version 3.0

Ce projet fournit un pipeline complet et reproductible de classification d'images alimentaires a partir d'un dataset construit avec OpenFoodFacts. La version `3.0` inclut :

- un dataset brut versionne avec DVC
- un dataset prepare `train/val/test` versionne avec DVC
- un pipeline d'entrainement PyTorch finalise
- un modele entraine versionne avec DVC
- des figures sauvegardees pendant l'evaluation

## Objectif

L'objectif du projet est de produire une version stable et reproductible d'un pipeline d'apprentissage supervisé pour la classification d'images. Le workflow complet suit ces etapes :

1. collecte des images dans `data/raw/`
2. preparation d'un split reproductible dans `data/processed/`
3. entrainement d'un modele `ResNet18`
4. sauvegarde du meilleur modele
5. versionnement des donnees et du modele avec DVC

## Structure du projet

```text
bidabi-clone-alone/
├── data/
│   ├── raw/                  # dataset source versionne avec DVC
│   ├── processed/            # split train/val/test versionne avec DVC
│   ├── raw.dvc
│   └── processed.dvc
├── models/
│   └── best_model_resnet18_finetuned.pth
├── reports/
│   └── figures/
├── src/
│   ├── prepare_dataset.py    # preparation du split reproducible
│   ├── classificator.py      # entrainement et evaluation
│   ├── data_loader.py
│   └── asyscrapper.py
├── requirements.txt
└── README.md
```

## Dataset utilise

Le dataset source de la version `3.0` est stocke dans `data/raw/` et suivi avec DVC.

Le dataset d'entrainement est construit a partir de :

```text
data/raw/images/
├── bread/
├── butter/
└── milk/
```

Le script `src/prepare_dataset.py` cree ensuite la structure suivante :

```text
data/processed/
├── train/
│   ├── bread/
│   ├── butter/
│   └── milk/
├── val/
│   ├── bread/
│   ├── butter/
│   └── milk/
└── test/
    ├── bread/
    ├── butter/
    └── milk/
```

Le split applique est :

- `train`: 70%
- `val`: 15%
- `test`: 15%

Le melange est reproductible grace a une seed fixe (`42`).

## Pipeline d'entrainement

Le script principal est `src/classificator.py`.

Il realise les etapes suivantes :

1. chargement du dataset pre-splite depuis `data/processed/`
2. application de data augmentation sur `train`
3. chargement d'un modele `ResNet18` pre-entraine
4. fine-tuning complet du modele
5. suivi de `train loss`, `validation loss` et `validation accuracy`
6. sauvegarde du meilleur modele selon la plus faible `validation loss`
7. evaluation sur le jeu de test
8. generation de figures d'analyse

### Data augmentation

Les transformations appliquees au train set sont :

- `Resize`
- `RandomHorizontalFlip`
- `RandomRotation`
- `RandomAffine`
- `ColorJitter`
- `GaussianBlur`
- `Normalize`

Les jeux `val` et `test` utilisent seulement :

- `Resize`
- `ToTensor`
- `Normalize`

### Critere de sauvegarde du meilleur modele

Le meilleur modele est choisi selon la plus petite `validation loss`.

Le fichier sauvegarde est :

```text
models/best_model_resnet18_finetuned.pth
```

### Early stopping

L'entrainement est configure pour un maximum de `20` epochs avec une patience de `3`.

Cela signifie :

- si la `validation loss` s'ameliore, le modele est sauvegarde
- si la `validation loss` ne s'ameliore plus pendant `3` epochs consecutives, l'entrainement s'arrete avant `20` epochs

## Figures generees

Pendant l'evaluation, les figures suivantes sont enregistrees dans `reports/figures/` :

- `training_validation_loss.png`
- `validation_accuracy.png`
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `roc_curves.png`
- `hardest_samples.png`
- `tsne_embedding.png`
- `umap_embedding.png` si `umap` est installe

## Reproduction du projet

### 1. Creer un environnement virtuel

```bash
python -m venv .venv
```

Sous PowerShell :

```bash
.venv\Scripts\Activate.ps1
```

### 2. Installer les dependances

```bash
pip install -r requirements.txt
```

### 3. Recuperer les donnees et le modele avec DVC

Le remote DVC du projet est un dossier local (`data/localstore`) : les images et le modele ne sont pas publies sur GitHub. Pour reproduire le projet, il faut reconstruire le dataset avec `src/asyscrapper.py` ou configurer son propre remote DVC.

Si les fichiers DVC sont presents et que le cache DVC est configure :

```bash
dvc pull
```

### 4. Generer le split d'entrainement

```bash
python src/prepare_dataset.py
```

### 5. Lancer l'entrainement

```bash
python src/classificator.py
```

## Versionnement

### Donnees

Les elements suivants sont versionnes avec DVC :

- `data/raw/`
- `data/processed/`
- `models/best_model_resnet18_finetuned.pth`

### Code

Le code source, la documentation et les fichiers `.dvc` sont versionnes avec Git.

## Fichiers principaux

- `src/prepare_dataset.py`: cree le split `train/val/test`
- `src/classificator.py`: entraine et evalue le modele
- `data/raw.dvc`: suivi DVC du dataset brut
- `data/processed.dvc`: suivi DVC du dataset prepare
- `models/*.dvc`: suivi DVC du modele entraine

## Resultat attendu

Une execution complete du pipeline doit produire :

- un dataset prepare dans `data/processed/`
- un meilleur modele dans `models/`
- des figures dans `reports/figures/`
- une version reproductible du projet prete pour le tag `v3.0`
