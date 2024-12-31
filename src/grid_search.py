import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

# Création du répertoire des résultats
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Chargement des données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Hyperparamètres pour GridSearch et RandomizedSearch
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30]
}
param_random = {
    "n_estimators": np.arange(50, 500, 50),
    "max_depth": np.arange(5, 35, 5)
}

# Configuration du modèle
model = RandomForestRegressor(random_state=42)

# Exécution de GridSearch
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_grid_params = grid_search.best_params_

# Exécution de RandomizedSearch
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_random,
    scoring="neg_mean_squared_error",
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
best_random_params = random_search.best_params_

# Validation des meilleurs résultats
best_model = RandomForestRegressor(**best_random_params, random_state=42)
best_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = best_model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des résultats
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_random_params, f)

# Création du rapport textuel
report = f"""
Meilleurs paramètres (GridSearch) : {best_grid_params}
Meilleurs paramètres (RandomizedSearch) : {best_random_params}

Performances sur l'ensemble de test :
- MSE : {mse:.4f}
- R² : {r2:.4f}
"""
with open("metrics/grid_search_report.txt", "w") as f:
    f.write(report)

print(report)
print("Les meilleurs paramètres et le rapport ont été sauvegardés.")
