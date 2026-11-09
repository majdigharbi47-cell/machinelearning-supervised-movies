# -*- coding: utf-8 -*-
"""
mesure_seuils.py - LECTURE SEULE
--------------------------------
Mesure l'effet d'un seuil minimal sur budget ET revenue dans
data_cleaned/financial_movies.csv (aucun fichier existant n'est modifie).

Pour chaque seuil, on affiche : films gardes, films retires,
et la repartition Hit / Flop restante.

A placer dans le dossier prevs (a cote de data_cleaned/), puis :
    python mesure_seuils.py
Rapport ecrit dans : results/audit/seuils.txt
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
FICHIER = BASE / "data_cleaned" / "financial_movies.csv"
SORTIE = BASE / "results" / "audit"
SEUILS = [0, 1_000, 10_000, 100_000]

lignes = []


def log(texte=""):
    print(texte)
    lignes.append(str(texte))


def main():
    if not FICHIER.exists():
        log(f"ABSENT : {FICHIER}")
        return

    df = pd.read_csv(
        FICHIER, usecols=["budget", "revenue", "financial_target"]
    )
    total = len(df)
    log(f"Films dans financial_movies.csv : {total}\n")
    log("seuil | gardes | retires | % retires | Hit % | Flop %")

    for seuil in SEUILS:
        garde = df[(df["budget"] >= seuil) & (df["revenue"] >= seuil)]
        n = len(garde)
        if n == 0:
            log(f"{seuil:>7} | 0 | {total} | 100% | - | -")
            continue
        hit = (garde["financial_target"] == "Financial_Hit").mean() * 100
        log(
            f"{seuil:>7} | {n:>6} | {total - n:>7} | "
            f"{100 * (total - n) / total:>8.1f}% | "
            f"{hit:>5.1f} | {100 - hit:>5.1f}"
        )

    log("\n(seuil 0 = situation actuelle, sans filtre)")

    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "seuils.txt").write_text("\n".join(lignes), encoding="utf-8")
    log(f"\nRapport enregistre dans : {SORTIE / 'seuils.txt'}")


if __name__ == "__main__":
    main()
