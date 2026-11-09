# -*- coding: utf-8 -*-
"""
construire_recommandation.py
----------------------------
Prepare le systeme de recommandation (base sur le CONTENU des films).

Principe : chaque film est decrit par un texte (genres + mots-cles + resume +
langue). On transforme ce texte en nombres (TF-IDF), puis deux films sont
"proches" si leurs textes se ressemblent (similarite cosinus).

Le catalogue complet a plus d'un million de films : on garde les N films les
plus votes (par defaut 40 000), qui ont des descriptions fiables.

Lecture : data_cleaned/recommendation_catalog.csv (lu par morceaux).
Sorties : models/reco.joblib  et  models/reco_eval.json

    python construire_recommandation.py
"""

import json
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import machine_learning as ml

N_FILMS = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
FICHIER = ml.DATA_DIR / "recommendation_catalog.csv"
MODELS_DIR = ml.BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

COLONNES = [
    "id", "title", "release_year", "genres", "original_language", "overview",
    "keywords", "vote_average", "vote_count", "poster_path",
]


def liste(valeur):
    """'Action, Science Fiction' -> ['Action', 'Science Fiction']"""
    if pd.isna(valeur) or str(valeur).strip().lower() in {"", "unknown", "nan"}:
        return []
    return [p.strip() for p in re.split(r"[,|;]", str(valeur)) if p.strip()]


def texte_film(ligne):
    genres = [g.replace(" ", "_") for g in liste(ligne["genres"])]
    mots = [m.replace(" ", "_") for m in liste(ligne["keywords"])]
    resume = "" if pd.isna(ligne["overview"]) else str(ligne["overview"])
    langue = str(ligne["original_language"])
    # Les genres sont repetes 3 fois et les mots-cles 2 fois : ils pesent plus
    # que le resume dans la comparaison.
    return " ".join(genres * 3 + mots * 2 + [resume, f"lang_{langue}"])


def main():
    debut = time.time()
    if not FICHIER.exists():
        raise FileNotFoundError(f"Introuvable : {FICHIER}")

    # 1) Lecture par morceaux : on ne garde que les films les plus votes
    gardes = []
    for morceau in pd.read_csv(FICHIER, usecols=COLONNES, chunksize=200_000,
                               low_memory=False):
        morceau = morceau[morceau["vote_count"].fillna(0) >= 20]
        gardes.append(morceau.nlargest(N_FILMS, "vote_count"))
    films = (pd.concat(gardes).nlargest(N_FILMS, "vote_count")
             .drop_duplicates("id").reset_index(drop=True))
    print(f"Films gardes : {len(films)} (vote_count minimum : "
          f"{int(films['vote_count'].min())})")

    # 2) Texte de chaque film puis TF-IDF
    textes = films.apply(texte_film, axis=1)
    tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", min_df=3, max_df=0.5,
        max_features=40_000, sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b",
    )
    matrice = tfidf.fit_transform(textes)
    print(f"Matrice TF-IDF : {matrice.shape}")

    # 3) Controle simple : les voisins partagent-ils les genres du film ?
    rng = np.random.default_rng(ml.RANDOM_STATE)
    echantillon = rng.choice(len(films), size=min(500, len(films)), replace=False)
    genres = [set(liste(g)) for g in films["genres"]]

    def jaccard(a, b):
        return len(a & b) / len(a | b) if (a | b) else 0.0

    sims_ok, jac_reco, jac_hasard = [], [], []
    for i in echantillon:
        sim = (matrice @ matrice[i].T).toarray().ravel()
        sim[i] = -1
        top = np.argsort(-sim)[:10]
        for j in top:
            jac_reco.append(jaccard(genres[i], genres[j]))
            sims_ok.append(bool(genres[i] & genres[j]))
        for j in rng.choice(len(films), size=10, replace=False):
            jac_hasard.append(jaccard(genres[i], genres[j]))
    evaluation = {
        "films_dans_le_modele": int(len(films)),
        "part_voisins_avec_un_genre_commun": float(np.mean(sims_ok)),
        "jaccard_genres_recommandations": float(np.mean(jac_reco)),
        "jaccard_genres_au_hasard": float(np.mean(jac_hasard)),
    }
    print(json.dumps(evaluation, indent=2))

    # 4) Sauvegarde
    fiche = films[["id", "title", "release_year", "genres", "vote_average",
                   "vote_count", "overview", "poster_path"]].copy()
    fiche["overview"] = fiche["overview"].fillna("").astype(str).str.slice(0, 400)
    joblib.dump({"tfidf": tfidf, "matrice": matrice, "films": fiche},
                MODELS_DIR / "reco.joblib", compress=3)
    (MODELS_DIR / "reco_eval.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Termine en {(time.time() - debut) / 60:.1f} min.")


if __name__ == "__main__":
    main()
