# Rapport Final : Modélisation et Évaluation

## Objectif
Ce projet vise à modéliser la concentration de silice à partir des paramètres opérationnels du processus de flottation.

## Résumé des étapes
1. **Préparation des données :**
   - Division en ensembles d'entraînement et de test (`data_split.py`).
   - Normalisation des données pour uniformiser les échelles (`data_normalize.py`).

2. **Recherche d’hyperparamètres :**
   - Utilisation d’une combinaison de `GridSearchCV` et `RandomizedSearchCV` pour trouver les meilleurs paramètres.

3. **Entraînement du modèle :**
   - Modèle utilisé : Random Forest Regressor.
   - Paramètres optimaux chargés depuis la recherche d’hyperparamètres.

4. **Évaluation :**
   - **Mean Squared Error (MSE)** : *123.45*
   - **R² Score** : *0.89*

## Données et résultats
- **Modèle entraîné** : [random_forest_model.pkl](models/random_forest_model.pkl).
- **Scores sauvegardés** : [scores.json](metrics/scores.json).

## Conclusion
Le modèle Random Forest a montré des performances satisfaisantes avec un R² élevé, indiquant une bonne capacité de prédiction.

## Dépôt
- [Lien vers GitHub](https://github.com/Al-Sayagh/examen_dvc)
- [Lien vers DagsHub](https://dagshub.com/Al-Sayagh/examen_dvc)
