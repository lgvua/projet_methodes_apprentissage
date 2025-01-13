#!/usr/bin/env python
# coding: utf-8

# **Kobla LEGBEDJE** – kobla.legbedje.etu@univ-lille.fr
# 
# **Luigi VUACHET** – luigi.vuachet.etu@univ-lille.fr
# # **PROJET  Méthodes d’Apprentissage**

# #**1.1 Introduction**
# # Contexte du projet
# Le cancer du sein est l'une des formes les plus courantes de cancer chez les femmes dans le monde. Une détection précoce joue un rôle essentiel dans l'amélioration des chances de survie et de guérison des patientes. Grâce aux progrès technologiques et à la disponibilité croissante des données médicales, les méthodes d’analyse basées sur l’intelligence artificielle (IA) et les algorithmes de classification sont devenues des outils incontournables pour assister les médecins dans le diagnostic. Ces outils permettent d’améliorer la précision des détections tout en réduisant le temps nécessaire à l’analyse des résultats.
# 
# Ce projet s’inscrit dans cette dynamique en exploitant des données médicales spécifiques, notamment des mesures liées aux tumeurs (telles que le rayon de la tumeur, la texture, le périmètre et l’aire), afin de déterminer si une tumeur est bénigne (B) ou maligne (M).
# 
# **Objectif de l'analyse**
# 
# L'objectif principal de notre étude est de développer un modèle de classification capable de prédire le diagnostic d'une tumeur à partir d’un ensemble de variables issues de mesures de tumeurs. Cette étude repose sur l’utilisation de plusieurs algorithmes de classification afin d’identifier celui offrant les meilleures performances.
# 
# Les étapes de notre projet incluent :
# 
# **1.2 Statistiques descriptives**
# 
# **1.3 Visualisation des données et corrélations**
# 
# **2 Algorithme de classification**
# 
# **2.1 Préparation des données**
# 
# **2.2 Implémentation des algorithmes**
# 
# **2.3 Analyse globale des performances**
# 
# **2.4 Conclusion**

# # **Rapport du Projet : Classification des **Tumeurs****

# In[ ]:


# wdbc.data
from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from google.colab import drive


# In[ ]:


drive.mount('/content/drive')
data=pd.read_csv('/content/drive/MyDrive/data/wdbc.data')
data.head()


# Renommage des colonnes

# In[ ]:


url = "/content/drive/MyDrive/data/wdbc.data"
column_names = [
    "radius_1", "texture_1", "perimeter_1", "area_1", "smoothness_1", "compactness_1",
    "concavity_1", "concave_points_1", "symmetry_1", "fractal_dimension_1",
    "radius_2", "texture_2", "perimeter_2", "area_2", "smoothness_2", "compactness_2",
    "concavity_2", "concave_points_2", "symmetry_2", "fractal_dimension_2",
    "radius_3", "texture_3", "perimeter_3", "area_3", "smoothness_3", "compactness_3",
    "concavity_3", "concave_points_3", "symmetry_3", "fractal_dimension_3"
]

# Charger les données avec les noms modifiés
columns = ["ID", "Diagnosis"] + column_names


data = pd.read_csv(url, header=None, names=columns)

# Description des données
data.drop('ID', axis=1, inplace=True)
print(data.head())
print(data.info())


# In[ ]:


# Analyse des classes
class_counts = data['Diagnosis'].value_counts()
print("Répartition des classes :\n", class_counts)
sns.countplot(data=data, x='Diagnosis')
plt.title("Distribution des classes")
plt.show()


# # **1.2 Statistiques descriptives**
# # Analyse descriptive détaillée
# **Statistiques globales :**
# Classe B (Bénigne) : 357 occurrences
# Classe M (Maligne) : 212 occurrences
# 
# **Analyse de la Répartition**
# 
# L'analyse met en évidence un déséquilibre important entre les deux classes : la classe B (Bénigne) représente environ 63 % des données, tandis que la classe M (Maligne) en constitue environ 37 %. Ce déséquilibre risque d'affecter les performances des algorithmes de classification en introduisant un biais en faveur de la classe majoritaire (B), au détriment de la classe minoritaire (M).
# 
# Faut-il rééquilibrer les classes ?
# 
# Oui, il est nécessaire d'adopter des méthodes de rééquilibrage pour garantir de meilleures performances des algorithmes sur les deux classes. Nous proposons d'utiliser l'approche du sur-échantillonnage (Oversampling), qui consiste à augmenter artificiellement la proportion de la classe minoritaire (M). Pour ce faire, nous utiliserons la technique SMOTE (Synthetic Minority Oversampling Technique), qui génère de nouveaux échantillons synthétiques à partir des données existantes de la classe minoritaire.

# In[ ]:


print(data.describe())


# # Analyse des Statistiques Descriptives
# 
# ## Tendances Centrales (Moyenne, Médiane, 50%)
# L'analyse des statistiques descriptives montre des tendances distinctes entre les différentes variables. Les variables à valeurs élevées : **Texture**, **périmètre** et **aire** présentent des moyennes nettement plus élevées que d'autres variables. Cela s'explique par le fait que ces variables mesurent des propriétés physiques globales et de grande échelle des tumeurs. Par exemple, **area_1** mesure des surfaces tumorales, avec une moyenne élevée en raison des tailles variées des tumeurs, tandis que **perimeter_1** capture la taille du contour des tumeurs, d'où des valeurs élevées par nature. Les variables à faibles valeurs : **Smoothness**, **compactness** et **concavity** ont des moyennes proches de zéro, ce qui suggère que ces variables sont sur une échelle plus fine. Elles capturent des propriétés relatives ou normalisées des tumeurs, comme la régularité, la compacité et la concavité des contours. Ces caractéristiques, bien que petites en valeur absolue, peuvent aussi être cruciales pour distinguer des tumeurs bénignes de malignes.
# 
# ## Variabilité (Écart-Type - std)
# L'analyse de la variabilité des données révèle des différences importantes entre les variables. Celles avec forte variabilité présentent des écarts-types élevés, indiquant une grande dispersion autour de la moyenne (**area_1** : écart-type = 351.91, **perimeter_1** : écart-type = 24.30). Les variables avec faible variabilité, comme **fractal_dimension_1**, **smoothness** ou **compactness**, présentent des écarts-types très faibles. Cela signifie que ces variables ont une distribution étroite autour de leur moyenne, ce qui témoigne d'une faible variabilité dans leurs valeurs.
# 
# ## Comparaison des Séries (1, 2, 3)
# Les caractéristiques des trois séries (1, 2, 3) montrent des tendances similaires en termes de moyennes et d'écarts-types. Cela suggère que chaque série représente des sous-groupes ou des angles d'analyse similaires des mêmes variables. Par exemple, **radius_1**, **radius_2** et **radius_3** mesurent probablement la même propriété (le rayon de la tumeur) mais sous des conditions ou des méthodes d'observation légèrement différentes.
# 

# # **1.3 Visualisation des données et corrélations**

# In[ ]:


# Corrélations entre variables
corr = data.iloc[:, 1:].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Matrice de corrélation")
plt.show()


# On observe une forte corrélation entre les variables d'un même type au sein des séries (1, 2, 3), ainsi qu’entre les variables
# **périmètre**, **rayon (radius)** et **aire (area)**. Cela s’explique par le fait que ces variables capturent des aspects étroitement liés des tumeurs. En effet, les colonnes associées à chaque variable montrent des valeurs très similaires, ce qui révèle une certaine redondance dans les données.
# 
# Par conséquent, l’analyse approfondie sera limitée aux 10 premières variables pour éviter toute surabondance inutile

# In[ ]:


# Etude poussée sur les correlation
#Calcule des corrélations entre les 10 premières caractéristiques.
feature = list(data.columns[1:11])


# In[ ]:



# Calcule de la matrice de corrélation
correlationData = data[feature].corr()

# Visualisation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlationData, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation entre les variables")
plt.show()


# **Corrélations fortes (positives et négatives)**
# 
# Les corrélations positives fortes concernent principalement les variables **radius_1**, **perimeter_1** et **area_1**, qui affichent des corrélations très élevées (proches de 1). Cela suggère qu'elles mesurent des propriétés similaires des tumeurs, notamment des aspects liés à leur taille ou à leurs dimensions globales.  
# 
# En revanche, **fractal_dimension_1** montre une corrélation négative avec les variables globales telles que **radius**, **area** ou **perimeter**. Cette variable semble donc fournir une information complémentaire et non redondante, ce qui pourrait la rendre particulièrement utile pour la classification des tumeurs.
# 
# Pour la suite, nous allons visualiser certaines mesures telles que **radius**, **texture**, **concavity** et **fractal_dimension** afin de comprendre leurs distributions et leurs relations avec la variable cible **Diagnosis**. Observons leurs histogrammes et les distributions associées par classe.
# 
# 

# In[ ]:



import seaborn as sns
import matplotlib.pyplot as plt

# les histogrammes pour une variable clé
sns.histplot(data=data, x="radius_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de radius_1 par classe")
plt.show()


# Pour la classe bénigne, la distribution de la variable **radius** tend à être centrée autour de valeurs plus faibles, tandis que pour la classe maligne, la distribution est déplacée vers des valeurs plus élevées, indiquant que les tumeurs malignes ont souvent un radius plus grand que les tumeurs bénignes.
# 
# Cela suggère que le **radius** est une variable discriminante importante pour séparer les deux classes.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# les histogrammes pour une variable clé
sns.histplot(data=data, x="texture_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de texture_1 par classe")
plt.show()


# La variable **texture** montre des distributions qui se chevauchent entre les classes bénignes et malignes. Il est observé que la classe maligne (M) présente des valeurs de texture plus élevées en moyenne. Bien que **texture** soit une variable utile pour différencier les deux classes, **radius** s'avère être plus discriminante.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Tracer les histogrammes pour une variable clé
sns.histplot(data=data, x="concavity_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de concavity_1 par classe")
plt.show()


# Pour la classe bénigne (B), la **concavité** est faible et centrée autour de valeurs proches de zéro, tandis que pour la classe maligne (M), on observe une plus grande dispersion vers des valeurs plus élevées, ce qui indique que les tumeurs malignes présentent souvent une **concavité** plus marquée.
# 
# Cette variable joue un rôle significatif dans la classification.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Tracer les histogrammes pour une variable clé
sns.histplot(data=data, x="fractal_dimension_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de fractal_dimension_1 par classe")
plt.show()


# La distribution de **fractal_dimension_1** présente des chevauchements importants entre les classes bénignes et malignes, mais on remarque que la classe maligne (M) tend à avoir des valeurs plus élevées. Cela indique que, bien que **fractal_dimension_1** ne soit pas une variable aussi discriminante que **radius**, elle conserve néanmoins un certain pouvoir informatif pour différencier les tumeurs.

# **Interprétation globlal**:
# Les histogrammes montrent que certaines variables, comme **radius** et **concavity**, présentent des différences nettes entre les classes, tandis que d'autres, comme **texture** et **fractal_dimension**, ont des distributions plus proches mais restent informatives.
# 
# Ces visualisations confirment que :  
# 
# - Les tumeurs malignes tendent à avoir des valeurs plus élevées pour des variables comme **radius** et **concavity**.  
# - La dispersion des valeurs est plus grande pour les tumeurs malignes, ce qui traduit une plus grande variabilité dans les mesures.  

# Traçons des pairplots pour observer les relations entre les 10 variables et leur impact sur la classification.

# In[ ]:


# Pairplot des premières variables
sns.pairplot(data, vars=["radius_1", "texture_1", "perimeter_1", "area_1", "smoothness_1", "compactness_1",
    "concavity_1", "concave_points_1", "symmetry_1", "fractal_dimension_1"], hue="Diagnosis", palette="coolwarm")
plt.show()


# On observe qu'une forte augmentation des variables **radius**, **perimeter** et **area** tend à favoriser la présence de la classe M. Ces variables restent très corrélées.
# 

# # Conclusion
# L’analyse de la matrice de corrélation révèle plusieurs points clés : une redondance importante entre des variables fortement corrélées (comme **radius**, **perimeter**, **area**) nécessite une réduction de dimensionnalité via des techniques telles que la PCA. Les variables faiblement corrélées, telles que **fractal_dimension** ou **smoothness**, présentent un potentiel discriminant élevé et doivent être conservées pour enrichir l’information utilisée dans les modèles de classification. Pour la suite, dans le cadre de la classification, nous allons utiliser une combinaison de réduction de dimension et de sélection de variables pertinentes afin de maximiser la performance des algorithmes de classification.
# 
# 

# # **2. Algorithmes de classification**
# # **2.1 Préparation des données**
# 2.1.1 **Division des données** :
# On sépare les données en train (70%) et test (30%).*
# 
# 2.1.2 **Prétraitement des données** :
# Avant d' appliquer les algorithme , les données seront centré et normalisé .
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Conversion de la variable cible
le = LabelEncoder()
data['Diagnosis'] = le.fit_transform(data['Diagnosis'])

# Séparation des données
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # **2.2 Implémentation des algorithmes**
# **1. Régression logistique**:
# C'est un modèle linéaire simple pour les problèmes de classification binaire (M ou B). Nous utiliserons la validation croisée pour ajuster les paramètres.

# 
# Tout d'abord, nous appliquerons l'algorithme de régression logistique sur nos données déséquilibrées afin d'observer l'impact de l'équilibrage des classes.
# 
# Ensuite, nous utiliserons l'Analyse en Composantes Principales (ACP) pour éliminer les variables redondantes, puis nous rééquilibrerons les classes. Nous appliquerons ensuite la régression logistique sur ces données transformées pour démontrer que le nouveau jeu de données offre de meilleurs résultats.
# 
# Enfin, nous testerons d'autres méthodes de classification sur les données projetées et comparerons leurs performances à l'aide de critères spécifiques.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Modèle de régression logistique
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Prédictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Évaluation
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob))


# **Régression logistique**
# 
# La régression logistique s'est montrée très efficace avec un AUC de 0.9975 et une excellente capacité de discrimination, avec un faible nombre de faux positifs (1) et de faux négatifs (4).
# 
# Nous obtenons une précision de 0.96 pour la classe bénigne et 0.98 pour la classe maligne. La régression logistique est particulièrement efficace pour prédire la classe maligne.
# 
# Nous allons appliquer la régression logistique à nos données, mais cette fois après réduction de la dimension (ACP).

# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Appliquer la PCA
pca = PCA(n_components=6)  # Laisser PCA calculer toutes les composantes
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Visualiser la variance expliquée
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.title('Variance expliquée par la PCA')
plt.show()


# À partir de ce graphique, nous observons que 6 composantes principales sont nécessaires pour expliquer 90 à 95 % de la variance.
# 
# Utilisons les projections des données d'entraînement et de test dans l'espace défini par les 6 premières composantes principales.

# In[ ]:


# Modèle de régression logistique
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_pca, y_train)

# Prédictions
y_pred = log_reg.predict(X_test_pca)
y_prob = log_reg.predict_proba(X_test_pca)[:, 1]

# Évaluation
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob))
#plot de la courbe roc


# On observe ici un AUC de 0.99824, ce qui est encore mieux qu'auparavant.
# 
# Équilibrons les données :
# 
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
# Rééquilibrage des données d'entraînement avec SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

# Modèle de régression logistique
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_resampled, y_train_resampled)

# Prédictions
y_pred = log_reg.predict(X_test_pca)
y_prob = log_reg.predict_proba(X_test_pca)[:, 1]

# Évaluation
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Modèle')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.legend(loc="lower right")
plt.title('Courbe ROC')
plt.legend()
plt.show()


# L'équilibre des classes n'a pas eu un grand effet sur notre modèle de régression logistique.
# 
# Nous allons utiliser les données projetées et rééquilibrées dans toutes les méthodes de classifications pour comparer nos modèles.

# **2. LDA (Linear Discriminant Analysis)**
# 
# LDA est adapté aux situations où les classes sont bien séparées et suivent des distributions gaussiennes. Il ne nécessite aucune calibration.
# 
# **3. QDA (Quadratic Discriminant Analysis)**
#     

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Modèle de discrimination linéaire
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_resampled, y_train_resampled)
# Modèle de discrimination quadratique
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_resampled, y_train_resampled)
lda_pred = lda.predict(X_test_pca)
qda_pred = qda.predict(X_test_pca)
# Évaluation
print("Matrice de confusion LDA :\n", confusion_matrix(y_test, lda_pred))
print("Rapport de classification LDA :\n", classification_report(y_test, lda_pred))
print("Matrice de confusion QDA :\n", confusion_matrix(y_test, qda_pred))
print("Rapport de classification QDA :\n", classification_report(y_test, qda_pred))
print("AUC LDA :", roc_auc_score(y_test, lda_pred))
print("AUC QDA :", roc_auc_score(y_test, qda_pred))


# In[ ]:


(y_test == lda_pred).mean()


# In[ ]:


(y_test == qda_pred).mean()


# **L'analyse discriminante linéaire (LDA)** a bien fonctionné pour les données équilibrées, offrant une précision parfaite pour la classe bénigne. Cependant, elle a montré ses limites pour la classe maligne, avec un rappel réduit (91 %) et 6 faux négatifs. Avec une AUC de 0.9531, cette méthode reste performante, mais elle est impactée par le déséquilibre des classes.
# 
# **L'analyse discriminante quadratique (QDA)** a offert des résultats équilibrés entre précision et rappel, avec une bonne performance globale. Elle a obtenu une précision de 97 % pour les tumeurs bénignes et de 95 % pour les tumeurs malignes, avec une AUC de 0.9625. QDA a ainsi démontré sa capacité à gérer des relations non linéaires.

# **4. Méthode des forêts aléatoires**

# La méthode des forêts aléatoires, basée sur des arbres de décision aléatoires, offre une classification efficace tout en gérant de manière optimale les interactions entre les variables.
# 
# Pour optimiser notre modèle, nous utiliserons la validation croisée avec GridSearchCV afin de déterminer les meilleurs hyperparamètres, notamment le nombre d'arbres (n_estimators) et la profondeur maximale (max_depth).

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#X_train_resampled, y_train_resampled
# Paramètres pour la grille
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_rf.fit(X_train_resampled, y_train_resampled)

# Meilleurs paramètres
print("Meilleurs paramètres :\n", grid_rf.best_params_)


# In[ ]:


# foret aléatoire avec les paramètre optimal
grid_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
grid_rf.fit(X_train_resampled, y_train_resampled)
# matrice de confusion
from sklearn.metrics import confusion_matrix
y_pred = grid_rf.predict(X_test_pca)
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
# rapport de classification
from sklearn.metrics import classification_report
print("Rapport de classification :\n", classification_report(y_test, y_pred))
# AUC
from sklearn.metrics import roc_auc_score
y_prob = grid_rf.predict_proba(X_test_pca)[:, 1]
print("AUC :", roc_auc_score(y_test, y_prob))


# In[ ]:


# Pour le bagginclassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# Modèle de bagging
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
param_grid = {'n_estimators': [10, 20, 30], 'max_samples': [0.5, 0.7, 0.9]}
grid_bagging = GridSearchCV(bagging, param_grid, cv=5, scoring='accuracy')
grid_bagging.fit(X_train_resampled, y_train_resampled)


# In[ ]:


# Meilleurs paramètres
print("Meilleurs paramètres :\n", grid_bagging.best_params_)


# In[ ]:


# on execute avec les paramètres optimales
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, max_samples=0.7, random_state=42)
bagging.fit(X_train_resampled, y_train_resampled)
print("Matrice de confusion :\n",sns.heatmap( confusion_matrix(y_test, y_pred), annot=True))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob))


# In[ ]:


# la courbe roc
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Modèle')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.legend(loc="lower right")
plt.title('Courbe ROC')
plt.legend()


# Les forêts aléatoires ont montré une très bonne performance globale grâce à leur robustesse face à la variance. Avec une précision de 95 % pour les deux classes et une AUC de 0.9940, elles démontrent une capacité équilibrée à classifier correctement les tumeurs bénignes et malignes, tout en gérant efficacement les interactions complexes entre les variables.
# 
# Nous allons ensuite utiliser la méthode **feature_importance**   des forêts aléatoires pour identifier les variables les plus influentes.

# In[ ]:


# utilise la méthode feature importance des forêts aléatoires pour identifier les variables les plus influentes
importances = grid_rf.feature_importances_
indices = np.argsort(importances)[::-1]
# Affichage des 10 premières variables les plus importantes
print("Les 3 premières variables les plus importantes sont :")
for i in range(3):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")


# Nous voyons que **radius** est la variable la plus importante dans la classification, ce qui confirme l'analyse que nous avons faite dans la première partie.

# **6. K-plus proches voisins (KNN)** : C'est une méthode simple basée sur la proximité des points. Pour le choix de K (nombre de voisins), nous utiliserons GridSearchCV pour déterminer le paramètre idéal et optimal.
# 

# In[ ]:


#Entraîner l’algorithme des k  plus proches voisin et évaluer sa qualité
#de classification sur les données test. on utilisera GridSearchCV pour le paramètre ideal
# des k.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# Paramètres pour la grille
param_grid = {'n_neighbors': [1,3, 5, 7, 9,11,20]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_knn.fit(X_train_resampled, y_train_resampled)
# Meilleurs paramètres
print("Meilleurs paramètres :\n", grid_knn.best_params_)





# In[ ]:


# entrainement avec k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_resampled, y_train_resampled)
# matrice de confusion
y_pred = knn.predict(X_test_pca)
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
# AUC
y_prob = knn.predict_proba(X_test_pca)[:, 1]
print("AUC :", roc_auc_score(y_test, y_prob))


# L’algorithme des k-plus proches voisins (KNN) a produit des résultats satisfaisants, avec une précision de 95% pour les deux classes et une AUC de 0.9687. Cependant, sa performance est légèrement inférieure à d’autres méthodes .

# **7.SVM (Support Vector Machines)**

# In[ ]:


# SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

clf = GridSearchCV(SVC(gamma='auto'), param_grid={'C': [1, 5, 10, 50]}, cv=5)
clf.fit(X_train_resampled, y_train_resampled)
print("Meilleurs paramètres :\n", clf.best_params_)


# In[ ]:


# test avec le meilleur partamètre
clf = SVC(C=1, gamma='auto')
clf.fit(X_train_resampled, y_train_resampled)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test_pca)
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_pred))


# In[ ]:


print(clf.score(X_test_pca, y_test))


# Le support vector machine (SVM) a montré de solides performances, en particulier pour des données bien séparables. Avec une précision de 95% pour les tumeurs bénignes et 97% pour les tumeurs malignes, et une AUC de 0.9515, SVM a démontré sa capacité à gérer des frontières complexes tout en maintenant un faible nombre d’erreurs.

# ## **2.3 Analyse globale des performances**
# 
# | **Algorithme**         | **Accuracy** | **AUC**   | **Classe bien classifiée** |
# |-------------------------|--------------|-----------|----------------------------|
# | Régression logistique  | 97%          | 0.9975    | Classe maligne             |
# | LDA                   | 96%          | 0.9531    | Classe bénigne             |
# | QDA                   | 96%          | 0.9625    | Équilibré                  |
# | Forêts aléatoires     | 95%          | 0.9940    | Équilibré                  |
# | KNN                   | 95%          | 0.9687    | Équilibré                  |
# | SVM                   | 96%          | 0.9515    | Équilibré                  |
# 

# 
# **Algorithme le plus adapté**
# 
# La **régression logistique** se distingue par son AUC élevé (0.9975) et sa capacité à bien classifier la classe maligne, ce qui est crucial dans un contexte médical. Elle combine simplicité, efficacité, et performance, en particulier sur des données linéairement séparables.
# Les forêts aléatoires et QDA offrent des performances comparables, avec une capacité robuste à gérer des données complexes et des interactions non linéaires. Elles sont recommandées si des relations non linéaires sont présentes.

# # **Conclusion Générale**
# 
# Dans ce projet, nous avons développé une solution robuste pour classifier les tumeurs bénignes et malignes à partir de données médicales. L'analyse descriptive a mis en évidence des variables discriminantes comme radius, area, et concave points, ainsi qu’un déséquilibre notable entre les classes, nécessitant une attention particulière lors de l’évaluation des modèles.
# 
# La régression logistique s’est révélée la méthode la plus performante, avec une AUC de 0.9975, combinant simplicité et fiabilité, en particulier pour prédire les tumeurs malignes. Les forêts aléatoires et QDA ont également offert de solides performances, adaptées à des données plus complexes.
# 
# Ce projet démontre l'importance d'une approche méthodique, combinant exploration des données et évaluation rigoureuse des modèles, pour répondre efficacement à des problématiques médicales critiques.
