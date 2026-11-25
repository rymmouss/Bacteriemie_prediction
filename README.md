# Prédiction de la bactériémie à partir d'un bilan sanguin

Projet de Machine Learning et Deep Learning appliqué à la santé (Master IDS — SciDoSanté, **Groupe 5**).  
L'objectif est de prédire la survenue d'une bactériémie dès la prise de sang standard à partir d'un bilan biologique, avant les résultats d'hémoculture qui prennent un à plusieurs jours.


## Équipe et répartition des rôles

Projet réalisé en équipe dans le cadre du Master IDS (Groupe 5) :
* **Ma contribution principale :** Développement du volet Deep Learning sous **PyTorch** (architecture réseau résiduel tabulaire, validation croisée stratifiée à 5 plis, gestion du déséquilibre des classes par pondération `pos_weight` dans `BCEWithLogitsLoss`, optimisation du seuil pour le F1-score et génération des prédictions).
* **Travail collectif :** Exploration et nettoyage physiologique des données (`Polars`), reconstruction des indices érythrocytaires, création des ratios cliniques, modélisation tabulaire (`Scikit-Learn`, Super-learner), suivi des expériences avec `MLflow` et présentation orale.


## Données et problématique

* **Jeu de données :** **11 753 patients** et **51 prédicteurs** (âge, sexe et 49 paramètres biologiques) issus de l'Hôpital général de Vienne (2006–2010), dans une version anonymisée.
* **Déséquilibre des classes :** Environ **8 %** de cas positifs. L'accuracy étant peu informative sur un tel déséquilibre, le **F1-score** (avec optimisation du seuil de décision) est la métrique principale, complétée par l'Average Precision et le ROC-AUC[cite: 17, 22, 23].


## Méthodologie

### 1. Nettoyage et ingénierie de variables (`Polars`)
* Conversion des valeurs physiologiquement impossibles en valeurs manquantes (`HCT <= 1`, `PLT == 0`, `WBC == 0`, `POTASS > 10`) plutôt que la suppression des lignes.
* Reconstruction de `RBC` et `HCT` à partir des indices érythrocytaires (`HGB / MCH * 10` et `HGB / MCHC * 100`), suivie du retrait des variables dérivées (`MCV`, `MCH`, `MCHC`).
* Ajout de ratios cliniques (`ratio_neu_lym`, `ratio_plt_lym`, `ratio_crp_alb`, `ratio_bun_crea`).

### 2. Préprocessing et modélisation
* **Découpage stratifié (80 % / 20 %) :** 9 402 patients pour l'entraînement et 2 351 patients pour le jeu de test interne.
* **Pipeline `Scikit-Learn` :** Transformation `log1p` sélective sur les variables asymétriques, imputation par la médiane avec indicateurs d'absence (`add_indicator=True`) et `RobustScaler`.
* **Modèles tabulaires (`Scikit-Learn` + `MLflow`) :** Régression logistique, forêt aléatoire, Extra Trees, boosting et **Super-learner** (`StackingClassifier`), avec recherche bayésienne des hyperparamètres et réglage du seuil.
* **Modèle Deep Learning (`PyTorch`) :** Réseau résiduel tabulaire (`TabularPyTorchModel` avec couche d'entrée de 128 neurones, 2 blocs résiduels avec `BatchNorm1d`, `SiLU` et `Dropout`), entraîné en 5 plis stratifiés avec `BCEWithLogitsLoss` pondérée (`pos_weight ≈ 11,5`), `AdamW` et `ReduceLROnPlateau`.


## Résultats clés

* **Validation croisée :** Le **Super-learner** atteint un F1 de **0,352** au seuil optimisé de **0,211** (contre 0,078 au seuil 0,5) et un ROC-AUC de **0,792**[cite: 17, 21, 22]. Le modèle **PyTorch** obtient un ROC-AUC de **0,735** et un F1 de **0,288** (seuil **0,705**).
* **Jeu de test interne (2 351 patients) :** F1 de **0,327**, précision de **0,300** (environ 3 alertes justes sur 10), rappel de **0,360** et ROC-AUC de **0,775**.
* **Prédictions finales :** Réentraînement sur les 11 753 patients (seuil recalculé à **0,198**) et export des prédictions (`ID`, `pred`) sur les **2 938 patients** du fichier de test (325 positifs prédits, soit **11,1 %**).


## Structure du dépôt

* `Script_Bacteriemie.ipynb` : Préprocessing Polars, modèles Scikit-Learn, optimisation et suivi MLflow
* `pytorch.ipynb` : Réseau de neurones résiduel sous PyTorch
* `Bacteriemie_presentation_groupe5.pdf` : Support de présentation orale
* `predictions_Bacteriemie.csv` : Fichier de prédictions (`ID`, `pred`)
