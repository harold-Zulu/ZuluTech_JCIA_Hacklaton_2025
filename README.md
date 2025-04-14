# Classificateur de Prunes Africaines

Ce projet implémente un modèle de classification d'images pour les prunes africaines en utilisant un réseau de neurones convolutif (CNN) avec PyTorch. Le modèle classe les prunes en six catégories : bonne qualité, non mûre, tachetée, fissurée, meurtrie et pourrie.

## Structure du Projet

```
.
├── src/                           # Code source
│   ├── plum_classifier.py         # Code d'entraînement du modèle
│   └── predict.py                 # Script de prédiction
├── dataset/                       # Dossier contenant les images de prunes
│   ├── unaffected/                # Prunes de bonne qualité
│   ├── unripe/                    # Prunes non mûres
│   ├── spotted/                   # Prunes tachetées
│   ├── cracked/                   # Prunes fissurées
│   ├── bruised/                   # Prunes meurtries
│   └── rotten/                    # Prunes pourries
├── app.py                         # Application Streamlit pour tester le modèle
├── plum_classifier.pth            # Modèle entraîné
├── training_history.png           # Graphiques d'accuracy et de loss
├── confusion_matrix.png           # Matrice de confusion
├── predictions.png                # Exemples de prédictions
├── requirements.txt               # Dépendances du projet
└── README.md                      # Documentation
```

## Installation

1. Clonez le dépôt :
```bash
git clone [URL_DU_REPO]
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Entraînement du modèle
1. Assurez-vous que vos images sont organisées dans le dossier `dataset` avec la structure suivante :
```
dataset/
├── unaffected/    # Prunes de bonne qualité
├── unripe/        # Prunes non mûres
├── spotted/       # Prunes tachetées
├── cracked/       # Prunes fissurées
├── bruised/       # Prunes meurtries
└── rotten/        # Prunes pourries
```

2. Exécutez le script d'entraînement :
```bash
python src/plum_classifier.py
```

### Interface de test
Pour tester le modèle avec une interface utilisateur :
```bash
streamlit run app.py
```

L'application Streamlit permet de :
- Télécharger une image de prune
- Obtenir la prédiction du modèle
- Voir les probabilités pour chaque catégorie

## Architecture du Modèle

Le modèle utilise une architecture CNN simple avec PyTorch :
- 3 couches de convolution avec max-pooling
- Une couche dense avec dropout
- Une couche de sortie softmax pour la classification

## Résultats

Les résultats de l'entraînement sont sauvegardés dans :
- `training_history.png` : Graphiques d'accuracy et de loss
- `confusion_matrix.png` : Matrice de confusion
- `plum_classifier.pth` : Le modèle entraîné

## Dépendances

- PyTorch 2.0.0+
- torchvision 0.15.0+
- NumPy 1.21.0+
- scikit-learn 1.0.0+
- Matplotlib 3.5.0+
- Seaborn 0.11.0+
- Pillow 9.0.0+
- Pandas 1.3.0+
- Streamlit (pour l'interface utilisateur) 