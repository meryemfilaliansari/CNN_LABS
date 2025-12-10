# Laboratoires de Réseaux de Neurones Convolutionnels (CNN)

**Auteur:** FILALI ANSARI Meryem  
**Date:** Décembre 2025

---

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Laboratoire 1 : CNN Fondamentaux et Convolution Simple](#laboratoire-1--cnn-fondamentaux-et-convolution-simple)
3. [Laboratoire 2 : CNN Appliqués et Détection de Contours](#laboratoire-2--cnn-appliqués-et-détection-de-contours)
4. [Dépendances et Installation](#dépendances-et-installation)
5. [Concepts Théoriques](#concepts-théoriques)
6. [Structure des Fichiers](#structure-des-fichiers)
7. [Références](#références)

---

## Vue d'Ensemble

Ce repository contient deux laboratoires pratiques explorant les **Réseaux de Neurones Convolutionnels (CNN)**, depuis les concepts fondamentaux jusqu'aux applications concrètes sur des images réelles :

- **Laboratoire 1** : Compréhension approfondie de la convolution avec construction manuelle d'un CNN simple
- **Laboratoire 2** : Application des CNN pour la détection de contours et traitement d'images réelles

Les notebooks démontrent une progression pédagogique du calcul mathématique manuel des convolutions jusqu'à l'utilisation de TensorFlow/Keras pour des architectures CNN complètes.

---

## Laboratoire 1 : CNN Fondamentaux et Convolution Simple

### Objectif

Comprendre en profondeur le mécanisme de convolution dans les CNN en :
- Construisant manuellement un input volume RGB (5x5x3)
- Créant des filtres de convolution 3x3x3
- Calculant étape par étape les feature maps résultantes
- Visualisant le processus complet avec stride=2 et padding='same'

### Dataset et Configuration

**Input Volume créé manuellement :**
- Dimensions : 5×5×3 (hauteur × largeur × canaux)
- 3 canaux RGB (Red, Green, Blue)
- Valeurs synthétiques pour faciliter la compréhension
- Format TensorFlow : (batch_size, height, width, channels)

**Exemple de l'Input Volume :**

```python
# Canal 1 (Red)
[[0, 0, 0, 0, 0],
 [0, 0, 1, 1, 0],
 [0, 1, 2, 1, 0],
 [0, 1, 1, 0, 0],
 [0, 0, 0, 0, 0]]

# Canal 2 (Green)
[[0, 0, 0, 0, 0],
 [0, 1, 2, 0, 0],
 [1, 2, 1, 0, 0],
 [1, 2, 1, 0, 0],
 [0, 1, 0, 0, 0]]

# Canal 3 (Blue)
[[0, 0, 2, 0, 0],
 [0, 0, 1, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [0, 1, 0, 0, 0]]
```

### Architecture du CNN

**Configuration du modèle :**

```python
model = keras.Sequential([
    layers.Input(shape=(5, 5, 3)),
    layers.Conv2D(
        filters=2,              # 2 filtres (feature maps en sortie)
        kernel_size=(3, 3),     # Taille 3×3 par canal
        strides=(2, 2),         # Stride = 2
        padding='same',         # Padding ajouté automatiquement
        use_bias=True,          # Avec biais
        activation=None         # Pas d'activation (convolution pure)
    )
])
```

**Paramètres détaillés :**
- **Filters** : 2 filtres de convolution
- **Kernel Size** : 3×3×3 (3×3 pixels sur 3 canaux)
- **Stride** : 2 (le filtre saute 2 pixels à chaque déplacement)
- **Padding** : 'same' (ajoute 1 pixel de padding pour conserver les dimensions)
- **Activation** : None (convolution pure sans fonction d'activation)
- **Bias** : Oui (un biais par filtre)

### Terminologie TensorFlow pour CNN

Le laboratoire 1 explique en détail la terminologie essentielle :

#### 1. INPUT SHAPE (input_shape)
```
Format : (height, width, channels)
Exemple : (5, 5, 3) = image 5×5 pixels avec 3 canaux RGB
```

#### 2. FILTERS (filters)
```
Nombre de filtres = nombre de feature maps en sortie
Chaque filtre détecte une caractéristique différente
Exemple : filters=2 → 2 feature maps en sortie
```

#### 3. KERNEL SIZE (kernel_size)
```
Taille du filtre de convolution
Format : (height, width)
Exemple : (3, 3) = filtre de 3×3 pixels
```

#### 4. STRIDES (strides)
```
Pas de déplacement du filtre
Format : (vertical_stride, horizontal_stride)
Exemple : (2, 2) = le filtre saute 2 pixels à chaque fois
```

#### 5. PADDING (padding)
```
'valid' : pas de padding → taille réduite
'same'  : padding ajouté → conserve la même taille (avec stride=1)
```

#### 6. ACTIVATION (activation)
```
Fonction d'activation après convolution
None = pas d'activation (convolution pure)
Autres : 'relu', 'sigmoid', 'tanh', etc.
```

#### 7. USE_BIAS (use_bias)
```
True = ajoute un biais b à chaque filtre
Formule : output = conv(input, weights) + bias
```

### Calcul de la Taille de Sortie

**Formule mathématique :**

```
Output_size = ((Input_size - Kernel_size + 2×Padding) / Stride) + 1
```

**Application à notre exemple :**

```
Input : 5×5×3
Kernel : 3×3×3
Stride : 2
Padding : same (= 1)
Filters : 2

Calcul hauteur :
Output_height = ((5 - 3 + 2×1) / 2) + 1 = ((5 - 3 + 2) / 2) + 1 = (4/2) + 1 = 3

Calcul largeur :
Output_width = ((5 - 3 + 2×1) / 2) + 1 = 3

Résultat : Sortie = 3×3×2
```

### Extraction et Visualisation des Poids

**Dimensions des poids (kernels) :**
```
Shape : (3, 3, 3, 2)
- Taille du filtre : 3×3
- Canaux d'entrée : 3 (RGB)
- Nombre de filtres : 2
```

**Dimensions des biais :**
```
Shape : (2,)
- Un biais par filtre
```

### Processus de Convolution Détaillé

Le laboratoire démontre le processus complet avec stride=2 :

**Positions de calcul (avec stride=2) :**
1. Position [0,0] : Coin supérieur gauche
2. Position [0,2] : Déplacement horizontal de 2 pixels
3. Position [2,0] : Déplacement vertical de 2 pixels
4. Position [2,2] : Coin inférieur droit accessible

**Calcul à chaque position :**

```python
Pour chaque position (y, x):
    1. Extraire région 3×3×3 de l'input
    2. Pour chaque canal c (0, 1, 2):
        conv_result += somme(région[:,:,c] * filtre[:,:,c])
    3. Ajouter le biais: conv_result += bias
    4. Placer le résultat dans feature_map[y//2, x//2]
```

**Formule mathématique de la convolution :**

```
Feature_Map[i,j] = Σ Σ Σ (Input[i*s + m, j*s + n, c] × Kernel[m, n, c]) + Bias
                   m n c

où:
- i, j : indices dans la feature map
- s : stride (2 dans notre cas)
- m, n : indices dans le kernel (0, 1, 2)
- c : indice du canal (0, 1, 2)
```

### Visualisations Générées

Le laboratoire 1 produit plusieurs visualisations pédagogiques :

#### 1. Input Volume (4 graphiques)
- Canal 1 (Red) : Heatmap 5×5
- Canal 2 (Green) : Heatmap 5×5
- Canal 3 (Blue) : Heatmap 5×5
- Vue RGB combinée : Visualisation couleur

#### 2. Filtres de Convolution
- Filtre W0 - 3 canaux : 3 heatmaps 3×3 + biais
- Filtre W1 - 3 canaux : 3 heatmaps 3×3 + biais
- Valeurs des poids affichées avec 2 décimales
- Colormap divergente (RdBu_r) centrée sur 0

#### 3. Feature Maps de Sortie
- Feature Map 0 : Résultat du filtre W0 (3×3)
- Feature Map 1 : Résultat du filtre W1 (3×3)
- Statistiques : Min, Max, Mean pour chaque feature map
- Colormap viridis pour visualisation

#### 4. Processus Complet de Convolution
- Grille 3×6 montrant toutes les étapes
- Ligne 1 : Input (3 canaux) + Filtre W0 (3 canaux)
- Ligne 2 : 4 positions de convolution avec stride=2
- Ligne 3 : Feature maps finales (2 filtres)
- Calculs manuels affichés pour chaque position

### Points Clés du Laboratoire 1

**Compréhension profonde :**
- Calcul manuel de chaque convolution
- Visualisation des poids et des feature maps
- Impact du stride sur la taille de sortie
- Rôle du padding dans la conservation des dimensions

**Formules essentielles :**
- Taille de sortie en fonction des paramètres
- Nombre de paramètres : (kernel_h × kernel_w × in_channels + 1) × n_filters
- Dans notre cas : (3 × 3 × 3 + 1) × 2 = 56 paramètres

**Observations :**
- Stride=2 réduit les dimensions de sortie de moitié
- Padding='same' avec stride=1 conserverait les dimensions
- Chaque filtre apprend à détecter une caractéristique différente
- Les biais permettent de décaler les activations

---

## Laboratoire 2 : CNN Appliqués et Détection de Contours

### Objectif

Appliquer les concepts de convolution à des cas pratiques de traitement d'image :
- Comprendre les filtres de détection de contours
- Implémenter des filtres classiques (Sobel, Prewitt, etc.)
- Appliquer les convolutions sur des images réelles
- Analyser les résultats visuellement

### Introduction aux CNN (Contexte Théorique)

Le laboratoire 2 commence par une introduction complète aux CNN :

#### Pourquoi les CNN ?

**Problèmes des réseaux entièrement connectés :**
- Nombre élevé de paramètres pour les images
- Perte de l'information spatiale locale
- Pas d'exploitation des relations entre pixels voisins

**Solutions apportées par les CNN :**
- Réduction du nombre de paramètres par partage des poids
- Préservation de l'information spatiale locale
- Capture des caractéristiques hiérarchiques

#### Principe de Base des Convolutions

Une convolution consiste à :
1. Appliquer un filtre (noyau) sur une image
2. Glisser le filtre sur toute l'image
3. Effectuer produit élément par élément + somme à chaque position
4. Créer une nouvelle image (feature map) avec les résultats

### Dataset et Images

**Image Synthétique Simple (5×5) :**

```python
image = np.array([
    [1, 2, 3, 4, 5],
    [5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5]
], dtype=np.float32)
```

**Image Réelle :**
- Chargée depuis Google Drive
- Convertie en niveaux de gris
- Format PIL → NumPy → TensorFlow
- Chemin : '/content/drive/MyDrive/.../maison2.jpg'

### Filtres de Détection de Contours

#### 1. Filtre de Détection de Bords Horizontaux

**Définition du filtre :**

```python
kernel_horizontal = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)
```

**Principe :**
- Ligne supérieure : poids négatifs (-1)
- Ligne centrale : poids nuls (0)
- Ligne inférieure : poids positifs (+1)

**Effet :**
- Détecte les transitions de clair à foncé (haut → bas)
- Réponse forte aux bords horizontaux
- Valeurs élevées = changement significatif

#### 2. Filtre de Détection de Bords Verticaux

```python
kernel_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)
```

**Principe :**
- Colonne gauche : poids négatifs
- Colonne centrale : poids nuls
- Colonne droite : poids positifs

**Effet :**
- Détecte les transitions de gauche à droite
- Réponse forte aux bords verticaux

#### 3. Filtre Sobel Horizontal

```python
kernel_sobel_h = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)
```

**Caractéristiques :**
- Version améliorée du filtre horizontal
- Poids centraux doublés (-2, +2)
- Plus sensible aux bords au centre
- Réduit le bruit en lissant

#### 4. Filtre Sobel Vertical

```python
kernel_sobel_v = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
```

#### 5. Filtre Prewitt

**Horizontal :**
```python
kernel_prewitt_h = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)
```

**Vertical :**
```python
kernel_prewitt_v = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)
```

#### 6. Filtre Laplacien

```python
kernel_laplacian = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)
```

**Principe :**
- Dérivée seconde
- Détecte tous les contours (toutes directions)
- Sensible au bruit
- Valeur centrale positive, voisins négatifs

### Application des Filtres avec TensorFlow

**Code général d'application :**

```python
# Reshape de l'image
image_tf = image.reshape((1, H, W, 1))

# Reshape du filtre
kernel_tf = kernel.reshape((3, 3, 1, 1))

# Application de la convolution
convolved = tf.nn.conv2d(
    image_tf, 
    kernel_tf, 
    strides=[1, 1, 1, 1],  # Stride de 1
    padding='VALID'         # Pas de padding
)

# Extraction du résultat
result = convolved.numpy().squeeze()
```

**Paramètres de tf.nn.conv2d :**
- **input** : Image au format [batch, height, width, channels]
- **filter** : Filtre au format [height, width, in_channels, out_channels]
- **strides** : [1, stride_h, stride_w, 1]
- **padding** : 'VALID' (sans padding) ou 'SAME' (avec padding)

### Calcul de la Taille de Sortie

**Avec padding='VALID' :**

```
Output_height = Input_height - Kernel_height + 1
Output_width = Input_width - Kernel_width + 1

Exemple avec image 5×5 et kernel 3×3 :
Output_height = 5 - 3 + 1 = 3
Output_width = 5 - 3 + 1 = 3
Sortie = 3×3
```

**Avec padding='SAME' :**

```
Output_height = ⌈Input_height / Stride⌉
Output_width = ⌈Input_width / Stride⌉

Exemple avec stride=1 :
Output conserve la même taille que l'input
```

### Visualisations Générées

Le laboratoire 2 produit plusieurs types de visualisations :

#### 1. Image Originale
- Affichage en niveaux de gris
- Colormap 'gray'
- Sans axes pour meilleure lisibilité

#### 2. Filtres (Kernels)
- Heatmap 3×3 de chaque filtre
- Valeurs annotées
- Colormap 'gray' ou divergente

#### 3. Images Convolutionnées
- Résultat de chaque filtre appliqué
- Comparaison côte à côte
- Mise en évidence des contours détectés

#### 4. Comparaisons Multiples
- Grid de sous-graphiques
- Image originale vs résultats de différents filtres
- Analyse comparative des détections

### Application sur Image Réelle

**Étapes du traitement :**

1. **Chargement** : 
   ```python
   img = Image.open(chemin_image).convert('L')
   image_real = np.array(img, dtype=np.float32)
   ```

2. **Préparation** :
   ```python
   image_tf = image_real.reshape((1, H, W, 1))
   ```

3. **Application des filtres** :
   - Horizontal edges
   - Vertical edges
   - Sobel (H et V)
   - Prewitt (H et V)
   - Laplacian

4. **Visualisation des résultats** :
   - Grid 2×4 ou 3×3
   - Original + 7 filtres différents
   - Analyse comparative

### Analyse des Résultats

**Observations typiques :**

1. **Filtre Horizontal** :
   - Détecte toits, fondations
   - Lignes horizontales accentuées
   - Transitions haut-bas

2. **Filtre Vertical** :
   - Détecte murs, colonnes
   - Lignes verticales accentuées
   - Transitions gauche-droite

3. **Sobel** :
   - Détection plus robuste
   - Moins de bruit que les filtres simples
   - Contours plus nets

4. **Prewitt** :
   - Similaire à Sobel
   - Légèrement moins de lissage
   - Sensibilité différente

5. **Laplacien** :
   - Détecte tous les contours
   - Plus sensible au bruit
   - Contours plus fins

### Comparaison des Filtres

| Filtre | Type | Directionnalité | Robustesse au Bruit | Usage |
|--------|------|-----------------|---------------------|-------|
| Horizontal Simple | Gradient | Horizontal uniquement | Faible | Pédagogique |
| Vertical Simple | Gradient | Vertical uniquement | Faible | Pédagogique |
| Sobel H | Gradient | Horizontal | Moyenne | Production |
| Sobel V | Gradient | Vertical | Moyenne | Production |
| Prewitt H | Gradient | Horizontal | Faible | Comparaison |
| Prewitt V | Gradient | Vertical | Faible | Comparaison |
| Laplacien | Dérivée 2nd | Toutes directions | Très faible | Détection fine |

### Points Clés du Laboratoire 2

**Applications pratiques :**
- Détection de contours dans images réelles
- Prétraitement pour vision par ordinateur
- Extraction de caractéristiques

**Concepts importants :**
- Différents types de filtres pour différentes détections
- Impact du padding sur la taille de sortie
- Trade-off sensibilité vs robustesse au bruit

**Compétences acquises :**
- Utilisation de tf.nn.conv2d
- Manipulation d'images avec PIL et NumPy
- Visualisation comparative avec Matplotlib
- Analyse qualitative des résultats

---

## Dépendances et Installation

### Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorflow>=2.8.0
Pillow>=8.0.0
```

### Installation

**Méthode 1 : pip**

```bash
pip install numpy matplotlib seaborn tensorflow pillow
```

**Méthode 2 : conda**

```bash
conda install numpy matplotlib seaborn tensorflow pillow
```

**Méthode 3 : requirements.txt**

```bash
pip install -r requirements.txt
```

### Vérification de l'Installation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Pillow version: {Image.__version__}")
```

### Configuration Google Colab

Pour le laboratoire 2 (accès Google Drive) :

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Environnement Recommandé

- **Python** : 3.8 ou supérieur
- **RAM** : Minimum 4 GB (8 GB recommandé)
- **GPU** : Non requis (CPU suffisant)
- **Système** : Windows 10/11, Linux, macOS
- **IDE** : Jupyter Notebook, JupyterLab, VS Code, Google Colab

---

## Concepts Théoriques

### Convolution 2D

**Définition mathématique :**

```
(I * K)[i, j] = Σ Σ I[i + m, j + n] × K[m, n]
                m n
```

où :
- I : Image d'entrée
- K : Kernel (filtre)
- [i, j] : Position dans l'image de sortie
- [m, n] : Position dans le kernel

**Propriétés :**
- Commutativité : I * K = K * I
- Associativité : (I * K₁) * K₂ = I * (K₁ * K₂)
- Distributivité : I * (K₁ + K₂) = I * K₁ + I * K₂
- Linéarité : a(I * K) = (aI) * K = I * (aK)

### Stride et Padding

**Stride (s) :**
- Pas de déplacement du filtre
- Stride grand → Output petit
- Stride=1 : déplacement pixel par pixel
- Stride=2 : déplacement tous les 2 pixels

**Padding (p) :**
- Ajout de pixels aux bords de l'image
- Padding=0 ('VALID') : pas d'ajout
- Padding='SAME' : ajout pour conserver la taille
- Calcul du padding : p = (K - 1) / 2 pour stride=1

**Formule générale de la taille de sortie :**

```
O = ⌊(I - K + 2p) / s⌋ + 1

où :
- O : Taille de l'output
- I : Taille de l'input
- K : Taille du kernel
- p : Padding
- s : Stride
- ⌊⌋ : Partie entière (floor)
```

### Nombre de Paramètres

**Pour une couche de convolution :**

```
Paramètres = (K_h × K_w × C_in + 1) × C_out

où :
- K_h, K_w : Dimensions du kernel
- C_in : Nombre de canaux d'entrée
- C_out : Nombre de filtres (canaux de sortie)
- +1 : Pour le biais (si use_bias=True)
```

**Exemple du Lab 1 :**
```
Paramètres = (3 × 3 × 3 + 1) × 2 = 28 × 2 = 56 paramètres
- Poids : 54
- Biais : 2
```

### Feature Maps

**Définition :**
- Résultat de l'application d'un filtre sur l'input
- Chaque filtre produit une feature map
- Les feature maps capturent différentes caractéristiques

**Hiérarchie des features :**
- Couches basses : Contours, textures simples
- Couches moyennes : Motifs, formes
- Couches hautes : Objets complexes, concepts

### Détection de Contours

**Gradient d'image :**

Le gradient mesure le taux de changement de l'intensité des pixels.

```
∇I = (∂I/∂x, ∂I/∂y)

Magnitude : |∇I| = √((∂I/∂x)² + (∂I/∂y)²)
Direction : θ = arctan(∂I/∂y / ∂I/∂x)
```

**Opérateurs de gradient :**

1. **Sobel** :
   - Approximation du gradient par convolution
   - Lissage intégré (poids 1-2-1)
   - Plus robuste au bruit

2. **Prewitt** :
   - Similaire à Sobel
   - Poids uniformes (1-1-1)
   - Plus simple mathématiquement

3. **Laplacien** :
   - Dérivée seconde
   - Détecte changements de gradient
   - Sensible au bruit

### Pourquoi les CNN Fonctionnent Bien

**1. Partage des Poids (Weight Sharing) :**
- Même filtre appliqué partout
- Réduit drastiquement le nombre de paramètres
- Invariance par translation

**2. Connexions Locales :**
- Chaque neurone connecté à région locale
- Exploite corrélation spatiale
- Champ récepteur (receptive field)

**3. Hiérarchie de Features :**
- Caractéristiques simples → complexes
- Composition de features
- Représentations abstraites

**4. Réduction Dimensionnelle :**
- Stride > 1 réduit la taille
- Pooling (non utilisé dans les labs)
- Compression progressive de l'information

---

## Structure des Fichiers

```
mssror/
├── CNN_LAB1.ipynb                    # Lab 1 : CNN Fondamentaux
├── CNN_LAB2.ipynb                    # Lab 2 : Détection de Contours
├── README.md                         # Ce fichier
├── .git/                             # Version control
└── images/                           # (optionnel) Images de test
    └── maison2.jpg                   # Image exemple pour Lab 2
```

---

## Exécution des Notebooks

### Jupyter Notebook

```bash
cd C:\Users\awati\Desktop\mssror
jupyter notebook
```

Puis ouvrir :
- `CNN_LAB1.ipynb`
- `CNN_LAB2.ipynb`

### VS Code

1. Ouvrir VS Code dans le répertoire
2. Installer l'extension Python et Jupyter
3. Ouvrir le fichier .ipynb
4. Sélectionner le kernel Python approprié
5. Exécuter les cellules séquentiellement

### Google Colab

1. Aller sur [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook
3. Sélectionner le fichier .ipynb
4. Pour Lab 2 : Monter Google Drive pour accéder aux images
5. Exécuter les cellules

---

## Applications Pratiques des CNN

### Vision par Ordinateur

**Classification d'images :**
- Reconnaissance d'objets (ImageNet)
- Classification médicale (radiographies)
- Identification de produits

**Détection d'objets :**
- YOLO (You Only Look Once)
- R-CNN, Fast R-CNN, Faster R-CNN
- SSD (Single Shot Detector)

**Segmentation d'images :**
- U-Net (segmentation biomédicale)
- Mask R-CNN
- DeepLab

### Traitement d'Images

**Amélioration d'images :**
- Super-résolution
- Débruitage
- Colorisation

**Style Transfer :**
- Transfert de style artistique
- Neural Style Transfer
- CycleGAN

### Applications Industrielles

**Inspection Qualité :**
- Détection de défauts
- Contrôle automatique
- Tri automatique

**Véhicules Autonomes :**
- Détection de piétons
- Reconnaissance de panneaux
- Segmentation de la route

**Sécurité :**
- Reconnaissance faciale
- Détection d'intrusion
- Surveillance vidéo

---

## Extensions Possibles

### Pour le Laboratoire 1

1. **Autres configurations** :
   - Tester stride=1, stride=3
   - Comparer padding='valid' vs 'same'
   - Augmenter le nombre de filtres

2. **Ajout d'activations** :
   - ReLU après convolution
   - Sigmoid, Tanh
   - Leaky ReLU, ELU

3. **Couches supplémentaires** :
   - MaxPooling2D
   - Plusieurs couches Conv2D
   - Batch Normalization

4. **Visualisations avancées** :
   - Animation du processus
   - Visualisation 3D
   - Champs récepteurs

### Pour le Laboratoire 2

1. **Filtres additionnels** :
   - Roberts Cross
   - Scharr
   - Canny edge detector complet

2. **Combinaisons de filtres** :
   - Gradient magnitude : √(Gx² + Gy²)
   - Direction du gradient : arctan(Gy/Gx)
   - Non-maximum suppression

3. **Applications avancées** :
   - Détection de coins (Harris)
   - Extraction de features SIFT
   - HOG (Histogram of Oriented Gradients)

4. **Dataset réel** :
   - CIFAR-10, CIFAR-100
   - MNIST, Fashion-MNIST
   - ImageNet

---

## Résultats et Conclusions

### Laboratoire 1

**Résultats clés :**
- Compréhension du calcul de convolution
- Visualisation des poids et feature maps
- Impact mesurable du stride et du padding
- Feature maps de dimension 3×3×2 générées avec succès

**Conclusion :** Le laboratoire 1 démontre que la convolution est une opération mathématique simple mais puissante. Le partage des poids permet de réduire drastiquement le nombre de paramètres tout en capturant efficacement les caractéristiques spatiales.

**Validation :**
- Calculs manuels correspondent aux résultats TensorFlow
- Feature maps cohérentes avec l'input et les filtres
- Formules de dimensionnement vérifiées

### Laboratoire 2

**Résultats clés :**
- Détection effective des contours horizontaux et verticaux
- Sobel plus robuste que les filtres simples
- Laplacien détecte tous les contours mais sensible au bruit
- Application réussie sur image réelle (maison)

**Conclusion :** Le laboratoire 2 illustre l'application pratique des CNN pour le traitement d'images. Les différents filtres révèlent différentes caractéristiques de l'image, base essentielle pour des tâches de vision par ordinateur plus complexes.

**Observations :**
- Choix du filtre dépend de l'application
- Preprocessing (normalisation) améliore les résultats
- Combinaison de filtres donne informations complémentaires

---

## Perspectives et Développements Futurs

### Court Terme

1. **Architectures classiques** :
   - LeNet-5 (reconnaissance de chiffres)
   - AlexNet (ImageNet 2012)
   - VGGNet (couches très profondes)

2. **Techniques modernes** :
   - ResNet (connexions résiduelles)
   - Inception (filtres multi-échelles)
   - MobileNet (efficacité mobile)

### Moyen Terme

1. **Transfer Learning** :
   - Utilisation de modèles pré-entraînés
   - Fine-tuning pour tâches spécifiques
   - Feature extraction

2. **Data Augmentation** :
   - Rotation, flip, zoom
   - Color jittering
   - Mixup, CutMix

### Long Terme

1. **Architectures avancées** :
   - Vision Transformers
   - EfficientNet
   - Neural Architecture Search

2. **Applications émergentes** :
   - Deepfakes
   - GANs pour génération d'images
   - Few-shot learning

---

## Références

### Livres

1. **Deep Learning with Python** (2nd Edition)  
   François Chollet (2021)  
   Manning Publications

2. **Deep Learning**  
   Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)  
   MIT Press

3. **Computer Vision: Algorithms and Applications** (2nd Edition)  
   Richard Szeliski (2022)  
   Springer

### Articles Fondateurs

1. **Gradient-Based Learning Applied to Document Recognition**  
   Y. LeCun, L. Bottou, Y. Bengio, P. Haffner (1998)  
   Proceedings of the IEEE  
   (LeNet-5)

2. **ImageNet Classification with Deep Convolutional Neural Networks**  
   Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)  
   NeurIPS  
   (AlexNet)

3. **Very Deep Convolutional Networks for Large-Scale Image Recognition**  
   Karen Simonyan, Andrew Zisserman (2014)  
   ICLR  
   (VGGNet)

4. **Deep Residual Learning for Image Recognition**  
   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)  
   CVPR  
   (ResNet)

### Documentation en Ligne

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Pillow Documentation](https://pillow.readthedocs.io/)

### Tutoriels et Cours

- [TensorFlow Tutorials - Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/cnn)
- [CS231n: Convolutional Neural Networks for Visual Recognition - Stanford](http://cs231n.stanford.edu/)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Datasets Populaires

- **MNIST** : Chiffres manuscrits (28×28, 10 classes)
- **CIFAR-10** : Objets naturels (32×32, 10 classes)
- **CIFAR-100** : Objets naturels (32×32, 100 classes)
- **ImageNet** : 1.4M images, 1000 classes
- **COCO** : Détection d'objets et segmentation

---

## Glossaire

**Activation** : Fonction non-linéaire appliquée après la convolution (ReLU, Sigmoid, Tanh).

**Batch Size** : Nombre d'échantillons traités simultanément.

**Bias (Biais)** : Terme constant ajouté au résultat de la convolution.

**Channel (Canal)** : Dimension de profondeur de l'image (RGB = 3 canaux).

**Convolution** : Opération mathématique appliquant un filtre sur une image.

**Feature Map** : Résultat de l'application d'un filtre sur l'input.

**Filter (Filtre)** : Également appelé kernel, matrice de poids apprise.

**Kernel** : Synonyme de filtre, matrice de convolution.

**Padding** : Ajout de pixels aux bords de l'image.

**Pooling** : Opération de réduction de dimension (non utilisée dans les labs).

**Receptive Field** : Région de l'input qui influence un neurone.

**Stride** : Pas de déplacement du filtre.

**Weight Sharing** : Utilisation des mêmes poids sur toute l'image.

---

## Auteur et Contact

**FILALI ANSARI Meryem**

Étudiante en Deep Learning et Vision par Ordinateur  


**Repository GitHub :** [LAB_LINEAR_AND_LOGISTIC_REG](https://github.com/meryemfilaliansari/LAB_LINEAR_AND_LOGISTIC_REG)

---

## Licence

Ce projet est à usage éducatif et pédagogique dans le cadre universitaire.

---

## Changelog

**Version 1.0 (Décembre 2025)**
- Laboratoire 1 : CNN Fondamentaux avec convolution manuelle
- Laboratoire 2 : Détection de contours et applications pratiques
- Documentation complète avec formules mathématiques
- Visualisations pédagogiques détaillées

---

**Dernière mise à jour :** Décembre 2025  
**Statut :** Complet et fonctionnel  
**Version Python :** 3.8+  
**Version TensorFlow :** 2.8+  
**Format :** Professionnel sans emojis ni icônes
