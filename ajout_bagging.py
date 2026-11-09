# -*- coding: utf-8 -*-
"""
ajout_bagging.py
----------------
Ajoute l'algorithme BAGGING au projet et dessine UN graphe qui compare tous
les algorithmes (Random Forest, Bagging, arbres, SVM, KNN, ...).

Principe du Bagging (en mots simples) :
  on entraine plusieurs arbres, chacun sur un echantillon different (tire avec
  remise) des films, puis ils votent. La difference avec Random Forest :
  Random Forest ajoute en plus un tirage au sort des variables a chaque
  coupure, ce qui rend les arbres plus differents entre eux.

Meme preparation des donnees, meme decoupage 80/20 (random_state=42) que
machine_learning.py : la comparaison est donc juste.

A lancer APRES machine_learning.py (il lit results/comparaison_modeles.csv) :
    python ajout_bagging.py

Sorties :
    results/bagging_resultats.csv
    results/comparaison_avec_bagging.csv       (split 80/20, tous les modeles)
    figures/machine_learning/comparaison_algos.png
    figures/machine_learning/ecart_train_test.png
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import machine_learning as ml
from outils_ml import MultiValueEncoder  # noqa: F401

BLUE, ORANGE, GREY = "#3B6FB6", "#C8641A", "#9AA3AE"
FICHIER_AVANT = ml.RESULTS_DIR / "comparaison_modeles.csv"


def bagging():
    arbre = DecisionTreeClassifier(class_weight="balanced",
                                   random_state=ml.RANDOM_STATE)
    try:  # scikit-learn >= 1.2
        return BaggingClassifier(estimator=arbre, n_estimators=200,
                                 n_jobs=-1, random_state=ml.RANDOM_STATE)
    except TypeError:  # anciennes versions
        return BaggingClassifier(base_estimator=arbre, n_estimators=200,
                                 n_jobs=-1, random_state=ml.RANDOM_STATE)


def entrainer_bagging():
    lignes = []
    for task, info in ml.TASKS.items():
        X, y = ml.load_task(task, info)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=ml.MAIN_TEST_SIZE, stratify=y,
            random_state=ml.RANDOM_STATE)
        debut = time.time()
        modele = ml.make_pipeline(X_tr, bagging())
        modele.fit(X_tr, y_tr)
        metriques, _ = ml.evaluate(modele, X_tr, y_tr, X_te, y_te)
        lignes.append({"task": task, "split": "80_20", "model": "Bagging",
                       **metriques, "best_parameters": "200 arbres complets",
                       "cv_f1_macro": float("nan")})
        print(f"  {info['titre']:<26} Bagging : F1-macro = "
              f"{metriques['f1_macro']:.3f}  accuracy = "
              f"{metriques['accuracy']:.3f}  ({time.time() - debut:.0f}s)")
    df = pd.DataFrame(lignes)
    df.to_csv(ml.RESULTS_DIR / "bagging_resultats.csv", index=False,
              encoding="utf-8-sig")
    return df


def dessiner(tout):
    dossier = ml.FIGURES_DIR
    dossier.mkdir(parents=True, exist_ok=True)
    taches = list(ml.TASKS.items())

    # ---- Graphe 1 : F1-macro de tous les modeles, une case par tache ----
    fig, axes = plt.subplots(1, len(taches), figsize=(5.2 * len(taches), 5.6),
                             sharex=True)
    for ax, (task, info) in zip(axes, taches):
        d = tout[tout["task"] == task].sort_values("f1_macro")
        meilleur = d["f1_macro"].max()
        couleurs = [ORANGE if v == meilleur else
                    GREY if "Baseline" in m else BLUE
                    for m, v in zip(d["model"], d["f1_macro"])]
        ax.barh(d["model"], d["f1_macro"], color=couleurs)
        for i, v in enumerate(d["f1_macro"]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(info["titre"])
        ax.set_xlim(0.3, max(0.9, meilleur + 0.08))
        ax.set_xlabel("F1-macro (test 20 %)")
    fig.suptitle("Comparaison de tous les algorithmes "
                 "(orange = meilleur, gris = baseline)")
    fig.tight_layout()
    fig.savefig(dossier / "comparaison_algos.png", dpi=200)
    plt.close(fig)

    # ---- Graphe 2 : ecart entre score train et score test (overfitting) ----
    fig, axes = plt.subplots(1, len(taches), figsize=(5.2 * len(taches), 5.6),
                             sharex=True)
    for ax, (task, info) in zip(axes, taches):
        d = tout[tout["task"] == task].sort_values("ecart_f1_macro_train_test")
        ax.barh(d["model"], d["ecart_f1_macro_train_test"], color=BLUE)
        for i, v in enumerate(d["ecart_f1_macro_train_test"]):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_title(info["titre"])
        ax.set_xlabel("F1 train - F1 test (petit = peu de par coeur)")
    fig.suptitle("Ecart entre apprentissage et test (overfitting)")
    fig.tight_layout()
    fig.savefig(dossier / "ecart_train_test.png", dpi=200)
    plt.close(fig)


def main():
    if not FICHIER_AVANT.exists():
        raise FileNotFoundError(
            f"{FICHIER_AVANT} introuvable : lance d'abord machine_learning.py")
    avant = pd.read_csv(FICHIER_AVANT)
    avant = avant[(avant["split"] == "80_20") & (avant["model"] != "Bagging")]

    print("Entrainement du Bagging ...")
    nouveau = entrainer_bagging()

    tout = pd.concat([avant, nouveau], ignore_index=True)
    colonnes = ["task", "model", "accuracy", "f1_macro", "recall_hit",
                "precision_hit", "train_f1_macro", "ecart_f1_macro_train_test"]
    tout[colonnes].round(4).to_csv(
        ml.RESULTS_DIR / "comparaison_avec_bagging.csv", index=False,
        encoding="utf-8-sig")

    dessiner(tout)
    print("\nClassement F1-macro (split 80/20) :")
    for task in ml.TASKS:
        d = tout[tout["task"] == task].sort_values("f1_macro", ascending=False)
        print(f"\n  {task}")
        for _, r in d.iterrows():
            print(f"    {r['model']:<30} {r['f1_macro']:.3f}")
    print(f"\nFigures : {ml.FIGURES_DIR}")


if __name__ == "__main__":
    main()
