import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Création du répertoire processed s'il n'existe pas
os.makedirs("data/processed", exist_ok=True)

# Chargement des données d'entraînement et de test
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Exclusion de la colonne non numérique
X_train = X_train.drop(columns=["date"])
X_test = X_test.drop(columns=["date"])

# Initialisation du scaler
scaler = StandardScaler()

# Normalisation des données
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarde des données normalisées
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

print("Les fichiers X_train_scaled.csv et X_test_scaled.csv ont été générés dans data/processed.")
