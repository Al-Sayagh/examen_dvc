import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

# Création des répertoires nécessaires
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Chargement des meilleurs paramètres depuis GridSearch/RandomizedSearch
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

print(f"Meilleurs paramètres chargés : {best_params}")

# Chargement des données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Entraînement du modèle avec les meilleurs paramètres
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Le modèle Random Forest a été entraîné et sauvegardé.")

# Évaluation du modèle sur le jeu de test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des scores dans un fichier JSON
scores = {
    "mean_squared_error": mse,
    "r2_score": r2
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(f"Scores sauvegardés dans 'metrics/scores.json': {scores}")
