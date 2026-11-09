# TMDB : prédire le succès d'un film (Hit / Flop) et recommander des films

Projet de Machine Learning sur le dataset TMDB (plus d'un million de films).

## Ce que fait le projet

| Partie | Question | Résultat (F1-macro, jeu de test) |
| --- | --- | --- |
| Financier avant la sortie | Le film rapportera-t-il au moins 2 fois son budget ? | environ 0,64 |
| Financier après la sortie | Même question, avec durée, popularité, note et votes | environ 0,75 |
| Public | Le film aura-t-il une note de 7 ou plus (au moins 100 votes) ? | environ 0,78 |
| Recommandation | Quels films ressemblent à celui-ci ? | TF-IDF + similarité cosinus |

Sept algorithmes sont comparés, face à une baseline : arbre de décision (en trois versions : complet, pré-élagué,
post-élagué), Random Forest, Bagging, régression logistique, SVM, KNN et Naive Bayes. Cela fait 10 modèles au total.
Le Random Forest est le modèle utilisé par l'application.
Le détail des choix et des limites est dans le dossier `docs/`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
```

## Les données

Le fichier `TMDB_movie_dataset_v11.csv` (dataset TMDB, à télécharger sur Kaggle) n'est **pas** dans le dépôt :
il est trop lourd. Place-le à la racine du projet, à côté des scripts.

## Ordre d'exécution

```bash
python nettoyage.py                    # 1. nettoie et crée data_cleaned/
python audit_projet.py                 # 2. vérifie la cible et l'absence de fuite de données
python mesure_seuils.py                #    (facultatif) mesure l'effet du seuil de 10 000 dollars
python eda.py                          # 3. graphiques d'exploration
python machine_learning.py             # 4. compare les modèles (long)
python ajout_bagging.py                # 5. ajoute le Bagging et le graphe comparatif
python analyse_variables.py            # 6. corrélation, importance, choix de K, tirages répétés
python entrainer_modeles_finaux.py     # 7. entraîne et sauvegarde les modèles dans models/
python construire_recommandation.py    # 8. prépare la recommandation
streamlit run app.py                   # 9. lance l'application
```

Les scripts doivent rester dans le même dossier : ils s'importent entre eux
(`machine_learning.py` et `outils_ml.py`) et cherchent les données à côté d'eux.

## Contenu du dépôt

```
nettoyage.py, audit_projet.py, mesure_seuils.py, eda.py     préparation et contrôle des données
outils_ml.py                                                 classe MultiValueEncoder (utilisée par les modèles sauvegardés)
machine_learning.py, ajout_bagging.py, analyse_variables.py  comparaison des modèles et analyses
entrainer_modeles_finaux.py, construire_recommandation.py    modèles pour l'application
app.py                                                       application Streamlit
results/   tableaux de résultats (CSV)
figures/   graphiques
docs/      rapports (nettoyage, machine learning, application, explication du code)
```

Non versionnés (voir `.gitignore`) : les données (`data_cleaned/`, fichier TMDB) et les modèles (`models/`).
Ils se régénèrent avec les scripts ci-dessus.

## Limites

- Les modèles financiers portent sur environ 10 000 films dont le budget et le revenu sont connus.
- « Après la sortie » estime le succès avec ce que le public a déjà fait, il ne le prédit pas à l'avance.
- Les probabilités affichées par l'application sont des estimations, pas des garanties.
- La recommandation se base sur le contenu des films (genres, mots-clés, résumé), pas sur les goûts des utilisateurs.
