import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
# Charger les données
chemin_fichier_csv = r'C:\Users\Admin\Desktop\Projet ML\milk\milknew.csv'
milk = pd.read_csv(chemin_fichier_csv)

# Afficher les premières lignes du DataFrame
print("les données de milk_data :\n", milk.head())

# Obtenez des informations sur le milkdata
print(milk.info())

# Afficher la description du jeu de données
desc = milk.describe()
print("la description du milk_data : ", desc)

# Création de l'histogramme pour le niveau de pH dans la base de données "milk"
plt.figure(figsize=(8, 6))
plt.hist(milk['pH'], bins=30, edgecolor='k', alpha=0.7, color='skyblue')
plt.xlabel("Niveau de pH")
plt.ylabel("Fréquence")
plt.title("Histogramme de la répartition des niveaux de pH dans la base de données Milk")
plt.show()
''
# Afficher la répartition du nombre d'exemples pour chaque grade en utilisant un graphique.
plt.figure(figsize=(8, 6))
sns.countplot(x="Grade", data=milk)
plt.xlabel("Catégories de Grade")
plt.ylabel("Fréquence")
plt.title("Distribution de probabilité des catégories de Grade")
plt.show()

# Afficher la répartition du nombre d'exemples pour chaque grade en fonction de la variable Odor en utilisant un graphique.
sns.countplot(x='Grade', hue='Odor', data=milk)
plt.title("Répartition des Milks par grade et Odor")
plt.show()

# Afficher la répartition du nombre d'exemples pour chaque grade en fonction de la variable Taste en utilisant un graphique.
sns.countplot(x='Grade', hue='Taste', data=milk)
plt.title("Répartition des Milks par grade et Taste")
plt.show()

# Vérification de la présence de valeurs nulles (NaN) dans le jeu de données "milk"
has_null_values_milk = milk.isnull().any().any()
print("Le jeu de données 'milk' contient des valeurs nulles (NaN) : ", has_null_values_milk)

# Vérification de la présence des duplications dans le jeu de données "milk"
duplicates = milk.duplicated()
# Affichage des lignes dupliquées
print("Nombre de lignes dupliquées : ", duplicates.sum())
print("Lignes dupliquées :\n  ")
print(milk[duplicates])
# Suppression des lignes dupliquées dans le jeu de données "milk"
milk_no_duplicates = milk.drop_duplicates()

# Affichage des informations après suppression des duplications
print("Nombre de lignes après suppression des duplications : ", len(milk_no_duplicates))

# Sélectionner uniquement les caractéristiques requises
selected_features = ["pH", "Temprature", "Taste", "Fat ", "Colour", "Grade"]
milk = milk[selected_features]
print(" les caractéristiques caractéristiques les plus pertinentes en fonction du jeu de données sont : \n ", milk.head())

# Affichez le contenu du vecteur Target du jeu de données.
target = milk.Grade
print(target)

# Transformez le jeu de données Milk en un DataFrame pour faciliter l'exploration.
milk_df = pd.DataFrame(milk, columns=selected_features)
print(milk_df)

# Divisez le jeu de données Milk en deux parties : 75% pour l'apprentissage et 25% pour le test en utilisant la fonction train_test_split().
X_train, X_test, y_train, y_test = train_test_split(milk_df, target, test_size=0.25, random_state=42)
print("Taille de l'ensemble d'apprentissage (X_train, y_train) :", X_train.shape, y_train.shape)
print("Taille de l'ensemble de test (X_test, y_test) :", X_test.shape, y_test.shape)

# Apprentissage supervisée : Méthode KNN
# Identifier les colonnes catégorielles et les colonnes numériques
categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(exclude=['object']).columns

# Create a preprocessor that scales numerical features and one-hot encodes categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Create a pipeline with the preprocessor and KNN classifier
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Train the KNN model
knn_pipeline.fit(X_train, y_train)
# Evaluate the performance of KNN
y_pred_knn = knn_pipeline.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of K-Nearest Neighbors (KNN): {:.2%}".format(accuracy_knn))
print("Classification Report for K-Nearest Neighbors (KNN):\n", classification_report(y_test, y_pred_knn))

# Apprentissage supervisée : Méthode SVM
# Create a pipeline with the preprocessor and SVM classifier
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', C=1.0))
])

# Train the SVM model
svm_pipeline.fit(X_train, y_train)
# Evaluate the performance of SVM
y_pred_svm = svm_pipeline.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of Support Vector Machine (SVM): {:.2%}".format(accuracy_svm))
print("Classification Report for Support Vector Machine (SVM):\n", classification_report(y_test, y_pred_svm))

# Évaluez les performances du modèle KNN en calculant la matrice de confusion, le rappel et la précision
y_pred_knn = knn_pipeline.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')

print("Matrice de Confusion pour le modèle KNN :\n", conf_matrix_knn)
print("Précision pour le modèle KNN: {:.2f}".format(precision_knn))
print("Rappel pour le modèle KNN: {:.2f}".format(recall_knn))
# Effectuer une évaluation du modèle en utilisant la validation croisée avec CV égal à 4
# pour le modèle KNN :
cv_scores_knn = cross_val_score(knn_pipeline, X_train, y_train, cv=4, scoring='precision_weighted')
print("Cross-Validation Scores pour KNN :", cv_scores_knn)
print("Average Score pour KNN:", np.mean(cv_scores_knn))
# Évaluez les performances du modèle SVM en calculant la matrice de confusion, le rappel et la précision
y_pred_svm = svm_pipeline.predict(X_test)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')

print("Matrice de Confusion pour le modèle SVM :\n", conf_matrix_svm)
print("Précision pour le modèle SVM: {:.2f}".format(precision_svm))
print("Rappel pour le modèle SVM: {:.2f}".format(recall_svm))

# Effectuer une évaluation du modèle en utilisant la validation croisée avec CV égal à 4
# pour le modèle SVM :
cv_scores_svm = cross_val_score(svm_pipeline, X_train, y_train, cv=4, scoring='precision_weighted')
print("Cross-Validation Scores pour SVM :", cv_scores_svm)
print("Average Score pour SVM :", np.mean(cv_scores_svm))
# Dernière question : calculer la matrice de corrélation
correlation = milk.corr()
print("la matrice de correlation est : \n ", correlation)

# Afficher la heatmap de la matrice de corrélation :
plt.figure(figsize=(7, 7))
sns.heatmap(correlation, annot=True, cmap='crest', linewidths=0.2)
plt.show()

# Afficher le nombre des lignes et des colonnes du dataframe (milk)
milk_shape = milk.shape
print("le nombre des lignes et des colonnes du milk", milk_shape)
#4
# Vérification de la présence de valeurs nulles (NaN) dans le jeu de données "milk"
has_null_values_milk = milk.isnull().any().any()
print("Le jeu de données 'milk' contient des valeurs nulles (NaN) : ", has_null_values_milk)
# Vérification de la présence des duplications dans le jeu de données "milk"
duplicates = milk.duplicated()
# Affichage des lignes duplicatées
print("Nombre de lignes dupliquées : ", duplicates.sum())
print("Lignes dupliquées :\n  ")
print(milk[duplicates])
# Suppression des lignes dupliquées dans le jeu de données "milk"
milk_no_duplicates = milk.drop_duplicates()

# Affichage des informations après suppression des duplications
print("Nombre de lignes après suppression des duplications : ", len(milk_no_duplicates))

