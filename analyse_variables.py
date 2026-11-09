# -*- coding: utf-8 -*-
"""
analyse_variables.py
--------------------
Complete le projet avec 4 analyses simples (aucun fichier existant modifie) :

  1. CORRELATION        : quelles variables numeriques bougent ensemble ?
  2. IMPORTANCE         : quelles variables comptent vraiment pour le modele ?
                          (methode "permutation" : on melange une colonne et on
                          regarde de combien le score baisse)
  3. CHOIX DE K (KNN)   : courbe du score selon K (pourquoi K=7)
  4. TIRAGE ALEATOIRE   : on refait le decoupage train/test 10 fois avec des
                          graines differentes -> moyenne +- ecart-type.
                          Si l'ecart-type est petit, le resultat ne depend pas
                          de la chance du tirage.

A placer dans prevs, a cote de machine_learning.py et outils_ml.py :
    python analyse_variables.py            (complet)
    python analyse_variables.py rapide     (test rapide, moins de repetitions)

Sorties : results/analyse/*.csv   et   figures/analyse/*.png
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import machine_learning as ml  # on reutilise les memes donnees et pretraitements

warnings.filterwarnings("ignore")

RAPIDE = len(sys.argv) > 1 and sys.argv[1].lower() == "rapide"
N_SEEDS = 3 if RAPIDE else 10
N_TREES = 60 if RAPIDE else 200
K_RANGE = [1, 3, 5, 7, 9, 11, 15, 21, 31]

BLUE, ORANGE = "#3B6FB6", "#C8641A"
OUT_RES = ml.RESULTS_DIR / "analyse"
OUT_FIG = ml.BASE_DIR / "figures" / "analyse"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def forest():
    return RandomForestClassifier(
        n_estimators=N_TREES, class_weight="balanced",
        random_state=ml.RANDOM_STATE, n_jobs=-1,
    )


# ------------------------------------------------------------
# 1. Correlation
# ------------------------------------------------------------
def correlation(task, info, X, y):
    num = X.select_dtypes(include="number").copy()
    num["cible (Hit=1)"] = (y == "Hit").astype(int).to_numpy()
    corr = num.corr(method="spearman")  # Spearman : robuste aux valeurs extremes
    corr.to_csv(OUT_RES / f"correlation_{task}.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation (Spearman)")
    ax.set_title(f"Correlations - {info['titre']}")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"correlation_{task}.png", dpi=200)
    plt.close(fig)

    lien = corr["cible (Hit=1)"].drop("cible (Hit=1)").sort_values(
        key=np.abs, ascending=False)
    print("  Correlation avec la cible :", lien.round(3).to_dict())


# ------------------------------------------------------------
# 2. Importance par permutation (par variable d'origine)
# ------------------------------------------------------------
def importance(task, info, X_tr, X_te, y_tr, y_te):
    model = ml.make_pipeline(X_tr, forest())
    model.fit(X_tr, y_tr)
    res = permutation_importance(
        model, X_te, y_te, scoring="f1_macro",
        n_repeats=3 if RAPIDE else 8, random_state=ml.RANDOM_STATE, n_jobs=1,
    )
    df = pd.DataFrame({
        "variable": X_te.columns,
        "baisse_f1_macro": res.importances_mean,
        "ecart_type": res.importances_std,
    }).sort_values("baisse_f1_macro", ascending=False)
    df.to_csv(OUT_RES / f"importance_permutation_{task}.csv",
              index=False, encoding="utf-8-sig")

    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.8))
    ax.barh(d["variable"], d["baisse_f1_macro"], xerr=d["ecart_type"],
            color=BLUE, ecolor="#555555")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Baisse du F1-macro quand on melange la variable")
    ax.set_title(f"Importance des variables - {info['titre']}")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"importance_permutation_{task}.png", dpi=200)
    plt.close(fig)
    print("  Top variables :", df.head(3)["variable"].tolist())


# ------------------------------------------------------------
# 3. Choix de K
# ------------------------------------------------------------
def courbe_k(task, info, X_tr, y_tr):
    ks = [k for k in K_RANGE if k < len(X_tr) * 0.5]
    cv = ml.make_cv()
    test_scores, train_scores = [], []
    for k in ks:
        pipe = ml.make_pipeline(X_tr, KNeighborsClassifier(n_neighbors=k), scale=True)
        s = cross_val_score(pipe, X_tr, y_tr, scoring="f1_macro", cv=cv, n_jobs=-1)
        test_scores.append(s.mean())
        pipe.fit(X_tr, y_tr)
        train_scores.append(f1_score(y_tr, pipe.predict(X_tr), average="macro"))
    pd.DataFrame({"K": ks, "cv_f1_macro": test_scores,
                  "train_f1_macro": train_scores}).to_csv(
        OUT_RES / f"courbe_K_{task}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, train_scores, "o-", color=ORANGE, label="Score sur l'entrainement")
    ax.plot(ks, test_scores, "o-", color=BLUE, label="Score en validation croisee")
    best = ks[int(np.argmax(test_scores))]
    ax.axvline(best, color="#888888", ls="--")
    ax.text(best, ax.get_ylim()[0], f" meilleur K = {best}", va="bottom", fontsize=9)
    ax.set_xlabel("K (nombre de voisins)")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"Choix de K - {info['titre']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"courbe_K_{task}.png", dpi=200)
    plt.close(fig)
    print(f"  Meilleur K sur la courbe : {best}")


# ------------------------------------------------------------
# 4. Tirage aleatoire repete
# ------------------------------------------------------------
def tirages(task, info, X, y):
    modeles = {
        "Arbre (profondeur 6)": lambda: (DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced",
            random_state=0), False),
        "Random Forest": lambda: (forest(), False),
        "Regression logistique": lambda: (LogisticRegression(
            max_iter=2000, class_weight="balanced"), True),
    }
    lignes = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
        for nom, fabrique in modeles.items():
            clf, scale = fabrique()
            pipe = ml.make_pipeline(X_tr, clf, scale=scale)
            pipe.fit(X_tr, y_tr)
            f1 = f1_score(y_te, pipe.predict(X_te), average="macro")
            lignes.append({"graine": seed, "modele": nom, "f1_macro": f1})
        print(f"    tirage {seed + 1}/{N_SEEDS} termine")
    df = pd.DataFrame(lignes)
    df.to_csv(OUT_RES / f"tirages_{task}.csv", index=False, encoding="utf-8-sig")
    resume = df.groupby("modele")["f1_macro"].agg(["mean", "std", "min", "max"])
    resume.to_csv(OUT_RES / f"tirages_resume_{task}.csv", encoding="utf-8-sig")
    print(resume.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    noms = list(modeles)
    data = [df[df["modele"] == n]["f1_macro"] for n in noms]
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(noms) + 1))
    ax.set_xticklabels(noms)
    for box in bp["boxes"]:
        box.set(facecolor=BLUE, alpha=0.6)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i), d, color=ORANGE, zorder=3, s=18)
    ax.set_ylabel("F1-macro sur le test")
    ax.set_title(f"{N_SEEDS} tirages aleatoires differents - {info['titre']}")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"tirages_{task}.png", dpi=200)
    plt.close(fig)


def main():
    for task, info in ml.TASKS.items():
        X, y = ml.load_task(task, info)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=ml.MAIN_TEST_SIZE, stratify=y,
            random_state=ml.RANDOM_STATE)
        print("\n[1] Correlation");   correlation(task, info, X, y)
        print("[2] Importance");      importance(task, info, X_tr, X_te, y_tr, y_te)
        print("[3] Choix de K");      courbe_k(task, info, X_tr, y_tr)
        print("[4] Tirages repetes"); tirages(task, info, X, y)
    print(f"\nTermine. Resultats : {OUT_RES}\nFigures : {OUT_FIG}")


if __name__ == "__main__":
    main()
