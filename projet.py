from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

data=pd.read_csv('./wdbc.data')
data.head()


url = "./wdbc.data"
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


# Analyse des classes
class_counts = data['Diagnosis'].value_counts()
print("Répartition des classes :\n", class_counts)
sns.countplot(data=data, x='Diagnosis')
plt.title("Distribution des classes")
plt.show()


print(data.describe())

# Corrélations entre variables
corr = data.iloc[:, 1:].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Matrice de corrélation")
plt.show()


# Etude poussée sur les correlation
#Calcule des corrélations entre les 10 premières caractéristiques.
feature = list(data.columns[1:11])



# Calcule de la matrice de corrélation
correlationData = data[feature].corr()

# Visualisation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlationData, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation entre les variables")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# les histogrammes pour une variable clé
sns.histplot(data=data, x="radius_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de radius_1 par classe")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# les histogrammes pour une variable clé
sns.histplot(data=data, x="texture_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de texture_1 par classe")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Tracer les histogrammes pour une variable clé
sns.histplot(data=data, x="concavity_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de concavity_1 par classe")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Tracer les histogrammes pour une variable clé
sns.histplot(data=data, x="fractal_dimension_1", hue="Diagnosis", kde=True, palette="coolwarm")
plt.title("Distribution de fractal_dimension_1 par classe")
plt.show()

# Pairplot des premières variables
sns.pairplot(data, vars=["radius_1", "texture_1", "perimeter_1", "area_1", "smoothness_1", "compactness_1",
    "concavity_1", "concave_points_1", "symmetry_1", "fractal_dimension_1"], hue="Diagnosis", palette="coolwarm")
plt.show()

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

(y_test == lda_pred).mean()


(y_test == qda_pred).mean()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#X_train_resampled, y_train_resampled
# Paramètres pour la grille
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_rf.fit(X_train_resampled, y_train_resampled)

# Meilleurs paramètres
print("Meilleurs paramètres :\n", grid_rf.best_params_)

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

# Pour le bagginclassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# Modèle de bagging
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
param_grid = {'n_estimators': [10, 20, 30], 'max_samples': [0.5, 0.7, 0.9]}
grid_bagging = GridSearchCV(bagging, param_grid, cv=5, scoring='accuracy')
grid_bagging.fit(X_train_resampled, y_train_resampled)

# Meilleurs paramètres
print("Meilleurs paramètres :\n", grid_bagging.best_params_)

# on execute avec les paramètres optimales
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, max_samples=0.7, random_state=42)
bagging.fit(X_train_resampled, y_train_resampled)
print("Matrice de confusion :\n",sns.heatmap( confusion_matrix(y_test, y_pred), annot=True))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob))

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

# utilise la méthode feature importance des forêts aléatoires pour identifier les variables les plus influentes
importances = grid_rf.feature_importances_
indices = np.argsort(importances)[::-1]
# Affichage des 10 premières variables les plus importantes
print("Les 3 premières variables les plus importantes sont :")
for i in range(3):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")

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

# SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

clf = GridSearchCV(SVC(gamma='auto'), param_grid={'C': [1, 5, 10, 50]}, cv=5)
clf.fit(X_train_resampled, y_train_resampled)
print("Meilleurs paramètres :\n", clf.best_params_)

# test avec le meilleur partamètre
clf = SVC(C=1, gamma='auto')
clf.fit(X_train_resampled, y_train_resampled)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test_pca)
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_pred))

print(clf.score(X_test_pca, y_test))
