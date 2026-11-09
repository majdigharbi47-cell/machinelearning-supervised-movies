# -*- coding: utf-8 -*-
"""
entrainer_modeles_finaux.py
---------------------------
Entraine le modele final (Random Forest) pour chacune des 3 taches et le
sauvegarde pour l'application.

Pour chaque tache :
  1. on separe 80 % (apprentissage) / 20 % (test) et on mesure la qualite
     reelle du modele -> ces chiffres seront affiches dans l'application ;
  2. puis on re-entraine le meme modele sur 100 % des films, qui est celui
     que l'application utilise (plus de donnees = un peu meilleur).

Sorties (dossier models/) :
    financial_pre.joblib, financial_post.joblib, audience.joblib
    meta.json   (metriques de test + listes de choix pour les menus)

A placer dans prevs, a cote de machine_learning.py et outils_ml.py :
    python entrainer_modeles_finaux.py
"""

import json
import time
from collections import Counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import machine_learning as ml
from outils_ml import MultiValueEncoder  # noqa: F401  (necessaire pour joblib)

MODELS_DIR = ml.BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def foret():
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=ml.RANDOM_STATE, n_jobs=-1,
    )


def valeurs_frequentes(serie, n):
    """Les n valeurs les plus frequentes d'une colonne a valeurs multiples."""
    compteur = Counter()
    for v in serie.dropna():
        for item in set(MultiValueEncoder._split_values(v)):
            compteur[item] += 1
    return [k for k, _ in compteur.most_common(n)]


def main():
    debut = time.time()
    meta = {"taches": {}, "choix": {}}
    choix_remplis = False

    for task, info in ml.TASKS.items():
        X, y = ml.load_task(task, info)

        # 1) Qualite mesuree sur des films jamais vus
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=ml.RANDOM_STATE)
        essai = ml.make_pipeline(X_tr, foret())
        essai.fit(X_tr, y_tr)
        pred = essai.predict(X_te)
        meta["taches"][task] = {
            "titre": info["titre"],
            "features": list(X.columns),
            "n_films": int(len(X)),
            "part_hit": float((y == "Hit").mean()),
            "test": {
                "accuracy": float(accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro")),
                "precision_hit": float(precision_score(y_te, pred, pos_label="Hit")),
                "recall_hit": float(recall_score(y_te, pred, pos_label="Hit")),
            },
        }
        print(f"  {task} : F1-macro test = {meta['taches'][task]['test']['f1_macro']:.3f}")

        # 2) Modele final sur 100 % des films
        final = ml.make_pipeline(X, foret())
        final.fit(X, y)
        joblib.dump(final, MODELS_DIR / f"{task}.joblib", compress=3)

        # Listes de choix pour les menus de l'application (une seule fois)
        if not choix_remplis and "genres" in X.columns:
            meta["choix"] = {
                "genres": valeurs_frequentes(X["genres"], 30),
                "langues": X["original_language"].value_counts().head(25).index.tolist(),
                "pays": valeurs_frequentes(X["production_countries"], 30),
                "societes": valeurs_frequentes(X["production_companies"], 60),
                "annee_min": int(X["release_year"].min()),
                "annee_max": int(X["release_year"].max()),
            }
            choix_remplis = True

    (MODELS_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTermine en {(time.time() - debut) / 60:.1f} min. Fichiers dans : {MODELS_DIR}")


if __name__ == "__main__":
    main()
