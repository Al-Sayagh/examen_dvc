Rapport Final : Modélisation et Évaluation
Objectif
Ce projet vise à modéliser la concentration de silice à partir des paramètres opérationnels du processus de flottation. L'objectif est d'étudier l'impact de ces paramètres sur la concentration de silice et de fournir un modèle prédictif performant.

Résumé des étapes
1. Préparation des données
Division des données :
Les données ont été divisées en ensembles d'entraînement (X_train, y_train) et de test (X_test, y_test) à l’aide du script data_split.py.
Normalisation des données :
Les ensembles X_train et X_test ont été normalisés pour uniformiser les échelles, via le script data_normalize.py.
2. Recherche d’hyperparamètres
Techniques utilisées :
Une combinaison de GridSearchCV et RandomizedSearchCV a été employée pour explorer les hyperparamètres.
Résultats :
Meilleurs paramètres obtenus par GridSearch : {'max_depth': 10, 'n_estimators': 100}.
Meilleurs paramètres obtenus par RandomizedSearch : {'n_estimators': 450, 'max_depth': 10}.
3. Entraînement du modèle
Modèle utilisé : Random Forest Regressor.
Paramètres optimaux : Ceux issus du RandomizedSearch ont été appliqués pour entraîner le modèle via le script train_model.py.
4. Évaluation
Les performances du modèle ont été évaluées sur l'ensemble de test.
Résultats :
Mean Squared Error (MSE) : 0.7710
R² Score : 0.2297
Données et résultats
Modèle entraîné : random_forest_model.pkl
Meilleurs paramètres : best_params.pkl
Rapport des recherches : grid_search_report.txt
Scores du modèle : scores.json
Dépôts
GitHub : Lien vers le dépôt GitHub
DagsHub : Lien vers le dépôt DagsHub
Conclusion
Le modèle Random Forest a été entraîné et évalué avec succès. Bien que les résultats de R² indiquent une performance modeste (0.2297), cela pourrait être amélioré par :

Une exploration plus approfondie des hyperparamètres.
L'utilisation d'autres modèles ou techniques avancées de preprocessing.
Le workflow complet est suivi par DVC et reproductible, avec toutes les étapes (préparation, recherche, entraînement et évaluation) intégrées dans un pipeline structuré.