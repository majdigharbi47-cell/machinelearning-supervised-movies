# -*- coding: utf-8 -*-
"""
audit_projet.py - AUDIT EN LECTURE SEULE du projet TMDB
-------------------------------------------------------
Ce script NE MODIFIE AUCUN fichier existant. Il :
1) profile le dataset original par morceaux (chunks) ;
2) profile les CSV de data_cleaned ;
3) verifie la coherence original -> nettoye -> 3 taches ;
4) liste les resultats deja generes dans results/.
Il ecrit seulement un rapport : results/audit/audit_report.txt

Lancer depuis le dossier du projet (prevs) :
    python audit_projet.py
ou, si le fichier original a un autre nom/chemin :
    python audit_projet.py "C:\\chemin\\vers\\TMDB_movie_dataset_v11.csv"
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ORIGINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "TMDB_movie_dataset_v11.csv"
DATA = BASE / "data_cleaned"
RESULTS = BASE / "results"
OUT = RESULTS / "audit"
CHUNK = 200_000

NUM = ["budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"]
CLEANED_FILES = [
    "tmdb_cleaned.csv",
    "recommendation_catalog.csv",
    "financial_movies.csv",
    "pre_release.csv",
    "post_release.csv",
    "audience_success.csv",
    "audience_analysis.csv",
]
TARGETS = {
    "financial_movies.csv": "financial_target",
    "pre_release.csv": "financial_target",
    "post_release.csv": "financial_target",
    "audience_success.csv": "audience_target",
}

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def titre(text):
    log("\n" + "=" * 70)
    log(text)
    log("=" * 70)


def check(label, condition):
    log(f"[{'OK' if condition else 'PROBLEME'}] {label}")


def lire(name, colonnes=None):
    """Lit un CSV de data_cleaned (colonnes optionnelles)."""
    path = DATA / name
    if not path.exists():
        return None
    if colonnes is None:
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, usecols=lambda c: c in colonnes, low_memory=False)


# ------------------------------------------------------------
# A. Dataset original
# ------------------------------------------------------------
def audit_original():
    titre("A. DATASET ORIGINAL : " + ORIGINAL.name)
    if not ORIGINAL.exists():
        log(f"ABSENT : {ORIGINAL}")
        return

    n = 0
    miss = None
    dtypes = None
    zeros = Counter()
    status = Counter()
    adult = Counter()
    ids = []
    both_pos = small_budget = small_revenue = 0

    for chunk in pd.read_csv(ORIGINAL, chunksize=CHUNK, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.lower()
        if dtypes is None:
            dtypes = chunk.dtypes.astype(str)
        n += len(chunk)
        m = chunk.isna().sum()
        miss = m if miss is None else miss.add(m, fill_value=0)

        for c in NUM:
            if c in chunk:
                zeros[c] += int((pd.to_numeric(chunk[c], errors="coerce") == 0).sum())
        if "status" in chunk:
            status.update(chunk["status"].fillna("NaN").value_counts().to_dict())
        if "adult" in chunk:
            adult.update(chunk["adult"].astype(str).value_counts().to_dict())
        if "id" in chunk:
            ids.append(pd.to_numeric(chunk["id"], errors="coerce").to_numpy())

        if {"budget", "revenue"} <= set(chunk.columns):
            b = pd.to_numeric(chunk["budget"], errors="coerce")
            r = pd.to_numeric(chunk["revenue"], errors="coerce")
            both = (b > 0) & (r > 0)
            both_pos += int(both.sum())
            small_budget += int((both & (b < 1000)).sum())
            small_revenue += int((both & (r < 1000)).sum())

    log(f"Lignes : {n}   Colonnes : {len(dtypes)}")
    log("\nColonne | type | manquants | % manquants")
    for col in dtypes.index:
        mv = int(miss[col])
        log(f"  {col} | {dtypes[col]} | {mv} | {100 * mv / n:.2f}%")

    log("\nValeurs egales a 0 (souvent un 'manquant' deguise) :")
    for c in NUM:
        if c in zeros:
            log(f"  {c} : {zeros[c]} ({100 * zeros[c] / n:.2f}%)")

    log("\nStatus :")
    for k, v in status.most_common():
        log(f"  {k} : {v}")
    log("Adult :")
    for k, v in adult.most_common():
        log(f"  {k} : {v}")

    if ids:
        all_ids = np.concatenate(ids)
        log(f"\nDoublons sur id : {len(all_ids) - len(np.unique(all_ids))}")

    log(f"\nFilms avec budget > 0 ET revenue > 0 : {both_pos}")
    log(f"  dont budget < 1000  (suspect) : {small_budget}")
    log(f"  dont revenue < 1000 (suspect) : {small_revenue}")


# ------------------------------------------------------------
# B. CSV nettoyes
# ------------------------------------------------------------
def profile_cleaned(name):
    path = DATA / name
    if not path.exists():
        log(f"\n{name} : ABSENT")
        return
    n = 0
    miss = None
    dtypes = None
    target = Counter()
    tcol = TARGETS.get(name)
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        if dtypes is None:
            dtypes = chunk.dtypes.astype(str)
        n += len(chunk)
        m = chunk.isna().sum()
        miss = m if miss is None else miss.add(m, fill_value=0)
        if tcol and tcol in chunk:
            target.update(chunk[tcol].value_counts().to_dict())

    log(f"\n--- {name} : {n} lignes x {len(dtypes)} colonnes")
    for col in dtypes.index:
        mv = int(miss[col])
        log(f"  {col} | {dtypes[col]} | manquants={mv} ({100 * mv / n:.2f}%)")
    if target:
        log(f"  Cible {tcol} : {dict(target)}")


def audit_cleaned():
    titre("B. FICHIERS NETTOYES (data_cleaned)")
    if not DATA.exists():
        log(f"ABSENT : {DATA}")
        return
    for name in CLEANED_FILES:
        profile_cleaned(name)


# ------------------------------------------------------------
# C. Coherence
# ------------------------------------------------------------
def audit_coherence():
    titre("C. COHERENCE ORIGINAL -> NETTOYE -> TACHES")
    fin = lire("financial_movies.csv")
    pre = lire("pre_release.csv")
    post = lire("post_release.csv")
    aud = lire("audience_success.csv")
    rec = lire("recommendation_catalog.csv", {"id"})

    if fin is None or pre is None or post is None:
        log("Fichiers financiers manquants : verification impossible.")
    else:
        check("financial / pre / post ont le meme nombre de lignes",
              len(fin) == len(pre) == len(post))
        check("memes id dans financial, pre et post",
              set(fin["id"]) == set(pre["id"]) == set(post["id"]))
        check("aucun doublon d'id dans pre_release", not pre["id"].duplicated().any())
        check("aucune colonne 'revenue*' dans pre_release",
              not any(c.lower().startswith("revenue") for c in pre.columns))
        check("aucune colonne 'revenue*' dans post_release",
              not any(c.lower().startswith("revenue") for c in post.columns))
        check("cible Hit/Flop = (revenue >= 2*budget)",
              bool(((fin["revenue"] >= 2 * fin["budget"])
                    == (fin["financial_target"] == "Financial_Hit")).all()))
        log(f"  Colonnes pre  : {list(pre.columns)}")
        log(f"  Colonnes post : {list(post.columns)}")
        for c in ["runtime", "popularity", "vote_average", "vote_count"]:
            if c in post:
                log(f"  post.{c} : manquants = {int(post[c].isna().sum())}")
        log("\n  Valeurs suspectes dans financial_movies :")
        log(f"    budget < 1000  : {int((fin['budget'] < 1000).sum())}")
        log(f"    revenue < 1000 : {int((fin['revenue'] < 1000).sum())}")
        ratio = fin["revenue"] / fin["budget"]
        log(f"    revenue/budget > 1000 : {int((ratio > 1000).sum())}")
        log(f"    budget min/median/max : {fin['budget'].min():.0f} / "
            f"{fin['budget'].median():.0f} / {fin['budget'].max():.0f}")
        if rec is not None:
            check("tous les id financiers sont dans recommendation_catalog",
                  set(fin["id"]) <= set(rec["id"]))

    if aud is None:
        log("\naudience_success.csv manquant.")
    else:
        log("\n  AUDIENCE")
        log(f"  Colonnes : {list(aud.columns)}")
        check("vote_count >= 100 partout", bool((aud["vote_count"] >= 100).all()))
        check("vote_average sans valeur manquante", not aud["vote_average"].isna().any())
        check("cible Audience = (vote_average >= 7)",
              bool(((aud["vote_average"] >= 7)
                    == (aud["audience_target"] == "Audience_Hit")).all()))
        check("aucun doublon d'id", not aud["id"].duplicated().any())
        if rec is not None:
            check("id audience inclus dans recommendation_catalog",
                  set(aud["id"]) <= set(rec["id"]))
        log("  Colonnes a risque pour le ML :")
        for c in ["release_date", "keywords", "poster_path", "overview"]:
            if c in aud:
                log(f"    {c} : {aud[c].nunique()} valeurs distinctes "
                    f"sur {len(aud)} lignes, manquants={int(aud[c].isna().sum())}")
        if "keywords" in aud:
            exemple = aud["keywords"].dropna().astype(str).head(3).tolist()
            log(f"    exemples keywords : {exemple}")


# ------------------------------------------------------------
# D. Resultats deja generes
# ------------------------------------------------------------
def audit_resultats():
    titre("D. RESULTATS DEJA GENERES")
    if not RESULTS.exists():
        log(f"ABSENT : {RESULTS}")
        return
    fichiers = [p for p in RESULTS.rglob("*") if p.is_file() and OUT not in p.parents]
    log(f"Fichiers dans results/ : {len(fichiers)}")
    comp = RESULTS / "comparaison_modeles.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        log(f"comparaison_modeles.csv : {df.shape}")
        log(f"  taches : {sorted(df['task'].unique())}")
        log(f"  splits : {sorted(df['split'].unique())}")
        log(f"  modeles : {sorted(df['model'].unique())}")
    else:
        log("comparaison_modeles.csv : ABSENT")
    figs = BASE / "figures"
    if figs.exists():
        log(f"Figures (png) dans figures/ : {len(list(figs.rglob('*.png')))}")


def main():
    audit_original()
    audit_cleaned()
    audit_coherence()
    audit_resultats()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nRapport enregistre dans : {OUT / 'audit_report.txt'}")


if __name__ == "__main__":
    main()
