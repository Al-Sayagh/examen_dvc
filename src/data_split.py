import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Création du répertoire 'data/processed' s'il n'existe pas
os.makedirs('data/processed', exist_ok=True)

# Chargement des données
data = pd.read_csv('data/raw/raw.csv')  # Assurez-vous que le fichier existe dans ce chemin

# Séparation des caractéristiques (X) et de la variable cible (y)
X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
y = data.iloc[:, -1]   # Dernière colonne (silica_concentrate)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des ensembles dans le dossier processed
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Les fichiers X_train, X_test, y_train, y_test ont été générés dans data/processed.")
