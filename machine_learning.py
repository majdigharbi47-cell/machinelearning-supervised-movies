# -*- coding: utf-8 -*-
"""
Projet TMDB - Classification Hit / Flop  (VERSION 2)
====================================================

Trois taches independantes :
  1) financial_pre  : succes financier AVANT la sortie
  2) financial_post : succes financier APRES la sortie
  3) audience       : succes aupres du public (vote_average >= 7)

Ce que fait ce script
---------------------
- 7 algorithmes officiels (Baseline, Decision Tree, Random Forest,
  Logistic Regression, SVM, KNN, Gaussian Naive Bayes).
- 3 separations train/test : 80/20, 70/30, 60/40
  (random_state = 42, stratify = y).
- Decision Tree en 3 versions : complet, pre-elagage, post-elagage.
- K de KNN et parametres des arbres choisis par validation croisee
  sur le TRAIN uniquement. Le test sert seulement a l'evaluation finale.
- Pretraitement dans un Pipeline : appris uniquement sur le train.
- Meme regle d'equilibrage (class_weight="balanced") pour tous les modeles
  qui l'acceptent (arbres, Random Forest, Logistic Regression, SVM).
  KNN et Gaussian Naive Bayes n'ont pas cette option.
- Mesures : Accuracy, Precision, Recall, F1 (Hit et Flop), F1-macro,
  score sur le train (pour detecter le surapprentissage),
  validation croisee 5 plis (moyenne +- ecart-type) sur le split 80/20.

Lancer depuis le dossier du projet (prevs) :
    python machine_learning.py            (complet, plus long)
    python machine_learning.py rapide     (split 80/20 seulement, pour tester)

Dependances : pandas numpy scikit-learn matplotlib
"""

import re
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from outils_ml import MultiValueEncoder

warnings.filterwarnings("ignore")


# ============================================================
# 1. PARAMETRES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_cleaned"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "machine_learning"
CACHE_DIR = BASE_DIR / ".cache_ml"  # cache temporaire (supprime a la fin)

RANDOM_STATE = 42
SPLITS = [0.20, 0.30, 0.40]  # part du test
MAIN_TEST_SIZE = 0.20  # split principal (courbes, importance, CV)
K_VALUES = [1, 3, 5, 7, 10]
CV_FOLDS = 5
N_ALPHAS = 20  # nombre de valeurs de ccp_alpha testees (post-elagage)
DEPTHS_CURVE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]

if "rapide" in sys.argv[1:]:
    SPLITS = [0.20]

# Palette de couleurs coherente pour tout le projet.
COLOR_HIT = "#2E9E5B"
COLOR_FLOP = "#D6453D"
COLOR_BLUE = "#3B6FB6"

PRE_FEATURES = [
    "budget",
    "release_year",
    "release_month",
    "original_language",
    "genres",
    "production_companies",
    "production_countries",
]
POST_EXTRA_FEATURES = ["runtime", "popularity", "vote_average", "vote_count"]
AUDIENCE_FEATURES = [
    "release_year",
    "release_month",  # calculee a partir de release_date
    "original_language",
    "genres",
    "production_companies",
    "production_countries",
    "keywords",
    "runtime",
    "popularity",
    "vote_count",
]

TASKS = {
    "financial_pre": {
        "file": "pre_release.csv",
        "target": "financial_target",
        "features": PRE_FEATURES,
        "titre": "Financier - Pre-release",
    },
    "financial_post": {
        "file": "post_release.csv",
        "target": "financial_target",
        "features": PRE_FEATURES + POST_EXTRA_FEATURES,
        "titre": "Financier - Post-release",
    },
    "audience": {
        "file": "audience_success.csv",
        "target": "audience_target",
        "features": AUDIENCE_FEATURES,
        "titre": "Audience",
    },
}

# Colonnes a plusieurs valeurs : nombre maximal de categories gardees.
MULTI_VALUE = {
    "genres": 30,
    "production_countries": 20,
    "production_companies": 50,
    "keywords": 50,
}
# Variables tres asymetriques : on applique log(1 + x).
LOG_COLUMNS = ["budget", "popularity", "vote_count"]
CATEGORICAL = ["original_language"]

# Colonnes interdites comme variables explicatives (fuite de donnees).
FORBIDDEN = {
    "revenue",
    "revenue_budget_ratio",
    "financial_target",
    "audience_target",
    "id",
    "title",
    "overview",
    "poster_path",
}


# ============================================================
# 2. ENCODAGE DES COLONNES A VALEURS MULTIPLES
# ============================================================
# La classe MultiValueEncoder est dans outils_ml.py (meme dossier) :
# ainsi le modele sauvegarde pourra etre recharge par l'application.


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

def load_task(task_name, info):
    """Charge un CSV, construit X (variables) et y (Flop / Hit)."""
    path = DATA_DIR / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    # Cible : "Financial_Hit" / "Audience_Hit" -> Hit, sinon Flop.
    is_hit = df[info["target"]].astype(str).str.lower().str.contains("hit")
    y = pd.Series(np.where(is_hit, "Hit", "Flop"), index=df.index, name="target")

    # Audience : le mois de sortie est calcule a partir de release_date.
    if task_name == "audience":
        dates = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_month"] = dates.dt.month

    features = info["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes dans {path.name} : {missing}")

    # Garde-fous anti-fuite de donnees.
    interdit = [c for c in features if c in FORBIDDEN]
    if task_name == "audience" and "vote_average" in features:
        interdit.append("vote_average")  # elle definit la cible Audience
    if interdit:
        raise ValueError(f"Variables interdites (fuite) : {interdit}")

    X = df[features].copy()
    for column in X.columns:
        if column in MULTI_VALUE or column in CATEGORICAL:
            X[column] = X[column].fillna("Unknown").astype(str)
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")

    print(f"\n{'=' * 70}")
    print(f"TACHE : {info['titre']}   ({path.name})")
    print(f"Dimensions de X : {X.shape}")
    print("Variables :", list(X.columns))
    print("Cible :", y.value_counts().to_dict())
    return X, y


# ============================================================
# 4. PRETRAITEMENT ET PIPELINES
# ============================================================

def build_preprocessor(X, scale=False):
    """
    - nombres : log(1+x) si asymetrique, valeurs manquantes -> mediane
      (+ une colonne 0/1 "etait manquant"), standardisation si scale=True ;
    - langue : one-hot ;
    - genres / pays / societes / mots-cles : une colonne 0/1 par categorie.
    """
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    log_cols = [c for c in numeric if c in LOG_COLUMNS]
    plain_cols = [c for c in numeric if c not in LOG_COLUMNS]

    def numeric_pipeline(use_log):
        steps = []
        if use_log:
            steps.append(
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
            )
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if log_cols:
        transformers.append(("num_log", numeric_pipeline(True), log_cols))
    if plain_cols:
        transformers.append(("num", numeric_pipeline(False), plain_cols))

    cat_cols = [c for c in X.columns if c in CATEGORICAL]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                max_categories=15,
                sparse_output=False,
            ),
            cat_cols,
        ))

    for column in X.columns:
        if column in MULTI_VALUE:
            transformers.append((
                f"multi_{column}",
                MultiValueEncoder(max_categories=MULTI_VALUE[column], min_frequency=20),
                [column],
            ))

    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )


def make_pipeline(X, classifier, scale=False, memory=None):
    return Pipeline(
        [("prep", build_preprocessor(X, scale)), ("model", classifier)],
        memory=memory,
    )


def make_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def clean_params(params):
    return {k.replace("model__", ""): v for k, v in params.items()}


# ============================================================
# 5. ENTRAINEMENT DES MODELES
# ============================================================
# Chaque fonction renvoie : (modele entraine, parametres choisis en texte,
#                            score CV de la selection ou NaN)

def fit_simple(X_train, y_train, classifier, scale=False):
    model = make_pipeline(X_train, classifier, scale)
    model.fit(X_train, y_train)
    return model, "", np.nan


def fit_knn(X_train, y_train, memory):
    """K choisi par validation croisee sur le train uniquement."""
    pipe = make_pipeline(X_train, KNeighborsClassifier(), scale=True, memory=memory)
    k_values = [k for k in K_VALUES if k <= len(X_train)]
    search = GridSearchCV(
        pipe, {"model__n_neighbors": k_values},
        scoring="f1_macro", cv=make_cv(), n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    best_k = search.best_params_["model__n_neighbors"]
    return search.best_estimator_, f"K={best_k}", search.best_score_


def fit_pre_pruned(X_train, y_train, memory):
    """Pre-elagage : on limite la taille de l'arbre des le depart."""
    pipe = make_pipeline(
        X_train,
        DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, ccp_alpha=0.0
        ),
        memory=memory,
    )
    grid = {
        "model__max_depth": [2, 3, 4, 6, 8, 12],
        "model__min_samples_leaf": [1, 5, 20],
        "model__min_samples_split": [2, 20],
    }
    search = GridSearchCV(
        pipe, grid, scoring="f1_macro", cv=make_cv(), n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, str(clean_params(search.best_params_)), search.best_score_


def fit_post_pruned(X_train, y_train, memory):
    """
    Post-elagage : l'arbre pousse completement, puis on coupe les branches
    peu utiles (ccp_alpha > 0). Valeurs de ccp_alpha tirees du chemin
    d'elagage calcule sur le TRAIN. Renvoie aussi la courbe train / CV.
    """
    prep = build_preprocessor(X_train)
    X_t = prep.fit_transform(X_train)
    full_tree = DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )
    path = full_tree.cost_complexity_pruning_path(X_t, y_train)
    alphas = np.unique(path.ccp_alphas[path.ccp_alphas > 0])
    if len(alphas) > 1:
        alphas = alphas[:-1]  # la derniere valeur donne un arbre a 1 feuille
    if len(alphas) > N_ALPHAS:
        # On choisit des valeurs reparties regulierement le long de la
        # sequence d'elagage : de l'arbre presque complet jusqu'a l'arbre
        # tres coupe (et non reparties sur une echelle de valeurs).
        positions = np.linspace(0, len(alphas) - 1, N_ALPHAS).astype(int)
        alphas = alphas[positions]
    alphas = [float(a) for a in alphas]

    pipe = make_pipeline(
        X_train,
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        memory=memory,
    )
    search = GridSearchCV(
        pipe, {"model__ccp_alpha": alphas},
        scoring="f1_macro", cv=make_cv(), n_jobs=-1, refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    curve = pd.DataFrame({
        "type": "post_elagage",
        "parametre": [p["model__ccp_alpha"] for p in search.cv_results_["params"]],
        "train_f1_macro": search.cv_results_["mean_train_score"],
        "cv_f1_macro": search.cv_results_["mean_test_score"],
    })
    return (
        search.best_estimator_,
        str(clean_params(search.best_params_)),
        search.best_score_,
        curve,
    )


def depth_curve(X_train, y_train, memory):
    """Score train / validation croisee selon la profondeur (pour la figure)."""
    rows = []
    for depth in DEPTHS_CURVE:
        pipe = make_pipeline(
            X_train,
            DecisionTreeClassifier(
                max_depth=depth, class_weight="balanced", random_state=RANDOM_STATE
            ),
            memory=memory,
        )
        res = cross_validate(
            pipe, X_train, y_train, cv=make_cv(), scoring="f1_macro",
            return_train_score=True, n_jobs=-1,
        )
        rows.append({
            "type": "pre_elagage_profondeur",
            "parametre": depth,
            "train_f1_macro": res["train_score"].mean(),
            "cv_f1_macro": res["test_score"].mean(),
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(model, X_train, y_train, X_test, y_test):
    pred = model.predict(X_test)
    pred_train = model.predict(X_train)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_hit": precision_score(y_test, pred, pos_label="Hit", zero_division=0),
        "recall_hit": recall_score(y_test, pred, pos_label="Hit", zero_division=0),
        "f1_hit": f1_score(y_test, pred, pos_label="Hit", zero_division=0),
        "precision_flop": precision_score(y_test, pred, pos_label="Flop", zero_division=0),
        "recall_flop": recall_score(y_test, pred, pos_label="Flop", zero_division=0),
        "f1_flop": f1_score(y_test, pred, pos_label="Flop", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "train_accuracy": accuracy_score(y_train, pred_train),
        "train_f1_macro": f1_score(y_train, pred_train, average="macro", zero_division=0),
        "pred_hit_rate": float(np.mean(pred == "Hit")),
    }
    metrics["ecart_f1_macro_train_test"] = metrics["train_f1_macro"] - metrics["f1_macro"]

    final = model.named_steps["model"]
    if hasattr(final, "get_n_leaves"):
        metrics["n_feuilles"] = int(final.get_n_leaves())
        metrics["profondeur"] = int(final.get_depth())
    return metrics, pred


def slug(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def cv_robustness(model, X_train, y_train):
    res = cross_validate(
        clone(model), X_train, y_train, cv=make_cv(), scoring="f1_macro",
        return_train_score=True, n_jobs=1,
    )
    return {
        "cv_f1_macro_moyenne": res["test_score"].mean(),
        "cv_f1_macro_ecart_type": res["test_score"].std(),
        "cv_train_f1_macro_moyenne": res["train_score"].mean(),
    }


# ============================================================
# 7. FIGURES
# ============================================================

def draw_confusion(ax, cm, title):
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}\n({pct[i, j] * 100:.0f} %)",
                ha="center", va="center", fontsize=10,
                color="white" if pct[i, j] > 0.5 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flop", "Hit"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Flop", "Hit"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title, fontsize=9)


def save_confusion_grid(matrices, task_title, split_name):
    """Une seule image avec les matrices de tous les modeles."""
    n = len(matrices)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, matrices):
        draw_confusion(ax, cm, name)
    for ax in axes[n:]:
        ax.axis("off")
    train, test = split_name.split("_")
    fig.suptitle(f"{task_title} - matrices de confusion (train {train} % / test {test} %)",
                 fontsize=12)
    fig.tight_layout()
    folder = FIGURES_DIR / "grilles"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"grille_{task_title_slug(task_title)}_{split_name}.png", dpi=130)
    plt.close(fig)


def task_title_slug(title):
    return slug(title.lower())


def save_single_confusion(cm, task_name, task_title, split_name, model_name):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    draw_confusion(ax, cm, f"{task_title}\n{model_name} ({split_name})")
    fig.tight_layout()
    folder = FIGURES_DIR / "matrices"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{task_name}_{split_name}_{slug(model_name)}.png", dpi=140)
    plt.close(fig)


def save_pruning_figure(task_name, task_title, depth_df, alpha_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(depth_df["parametre"], depth_df["train_f1_macro"], "o--",
            color=COLOR_BLUE, label="Train")
    ax.plot(depth_df["parametre"], depth_df["cv_f1_macro"], "o-",
            color=COLOR_HIT, label="Validation croisée")
    ax.set_xlabel("Profondeur maximale de l'arbre (max_depth)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Pré-élagage : limiter la profondeur")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(alpha_df["parametre"], alpha_df["train_f1_macro"], "o--",
            color=COLOR_BLUE, label="Train")
    ax.plot(alpha_df["parametre"], alpha_df["cv_f1_macro"], "o-",
            color=COLOR_HIT, label="Validation croisée")
    ax.set_xscale("log")
    ax.set_xlabel("ccp_alpha (plus grand = arbre plus coupé)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Post-élagage : couper les branches inutiles")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"{task_title} - effet de l'élagage (train 80 % / test 20 %)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"elagage_{task_name}.png", dpi=140)
    plt.close(fig)


def save_importance(task_name, task_title, names, importances):
    table = pd.DataFrame({"variable": names, "importance": importances})
    table = table.sort_values("importance", ascending=False)
    table.to_csv(
        RESULTS_DIR / f"importance_variables_{task_name}.csv",
        index=False, encoding="utf-8-sig",
    )
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["variable"], top["importance"], color=COLOR_BLUE)
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title(f"{task_title} - 15 variables les plus importantes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"importance_{task_name}.png", dpi=140)
    plt.close(fig)


# ============================================================
# 8. UNE TACHE, UN SPLIT
# ============================================================

def run_experiment(task_name, info, X, y, test_size, extras):
    split_name = f"{round((1 - test_size) * 100)}_{round(test_size * 100)}"
    is_main = abs(test_size - MAIN_TEST_SIZE) < 1e-9
    out_dir = RESULTS_DIR / task_name / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n--- {info['titre']} | train/test = {split_name} "
          f"({len(X_train)} / {len(X_test)} lignes) ---")

    cache = str(CACHE_DIR)
    balanced = {"class_weight": "balanced"}

    # (nom, fonction d'entrainement)
    plan = [
        ("Baseline (ZeroR)",
         lambda: fit_simple(X_train, y_train, DummyClassifier(strategy="most_frequent"))),
        ("Decision Tree - complet",
         lambda: fit_simple(X_train, y_train,
                            DecisionTreeClassifier(random_state=RANDOM_STATE, **balanced))),
        ("Decision Tree - pré-élagage",
         lambda: fit_pre_pruned(X_train, y_train, cache)),
        ("Decision Tree - post-élagage",
         lambda: fit_post_pruned(X_train, y_train, cache)),
        ("Random Forest",
         lambda: fit_simple(X_train, y_train,
                            RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                   random_state=RANDOM_STATE, **balanced))),
        ("Logistic Regression",
         lambda: fit_simple(X_train, y_train,
                            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                               **balanced), scale=True)),
        ("SVM (RBF)",
         lambda: fit_simple(X_train, y_train,
                            SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE, **balanced),
                            scale=True)),
        ("KNN",
         lambda: fit_knn(X_train, y_train, cache)),
        ("Gaussian Naive Bayes",
         lambda: fit_simple(X_train, y_train, GaussianNB(), scale=True)),
    ]

    results, robustness, matrices = [], [], []

    for model_name, trainer in plan:
        start = time.time()
        fitted = trainer()
        model, best_params, cv_score = fitted[0], fitted[1], fitted[2]

        metrics, pred = evaluate(model, X_train, y_train, X_test, y_test)
        row = {
            "task": task_name, "split": split_name, "model": model_name,
            **metrics, "best_parameters": best_params, "cv_f1_macro": cv_score,
        }
        results.append(row)
        print(f"  {model_name:<30} acc={metrics['accuracy']:.3f}  "
              f"F1-macro={metrics['f1_macro']:.3f}  "
              f"Recall-Hit={metrics['recall_hit']:.3f}  "
              f"({time.time() - start:.0f}s)")

        cm = confusion_matrix(y_test, pred, labels=["Flop", "Hit"])
        matrices.append((model_name, cm))

        report = classification_report(
            y_test, pred, labels=["Flop", "Hit"], zero_division=0
        )
        (out_dir / f"rapport_{slug(model_name)}.txt").write_text(
            f"Tache : {info['titre']}\nModele : {model_name}\nSplit : {split_name}\n"
            f"Parametres choisis : {best_params}\n\n{report}",
            encoding="utf-8",
        )

        if is_main:
            save_single_confusion(cm, task_name, info["titre"], split_name, model_name)
            robustness.append({
                "task": task_name, "model": model_name,
                **cv_robustness(model, X_train, y_train),
            })
            if model_name == "Decision Tree - post-élagage":
                extras["alpha_curve"] = fitted[3]
            if model_name == "Random Forest":
                forest = model.named_steps["model"]
                try:
                    names = list(model.named_steps["prep"].get_feature_names_out())
                except Exception:
                    names = [f"x{i}" for i in range(len(forest.feature_importances_))]
                save_importance(task_name, info["titre"], names, forest.feature_importances_)

    save_confusion_grid(matrices, info["titre"], split_name)

    if is_main:
        print("  Courbe profondeur / score (pre-elagage)...")
        extras["depth_curve"] = depth_curve(X_train, y_train, cache)

    return results, robustness


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():
    start_all = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # On ne remplace jamais silencieusement un ancien tableau de resultats.
    old = RESULTS_DIR / "comparaison_modeles.csv"
    if old.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = RESULTS_DIR / f"comparaison_modeles_ancienne_{stamp}.csv"
        shutil.move(str(old), str(backup))
        print(f"Ancien tableau conserve : {backup.name}")

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results, all_robustness, failed = [], [], []

    try:
        for task_name, info in TASKS.items():
            try:
                X, y = load_task(task_name, info)
                extras = {}
                for test_size in SPLITS:
                    res, rob = run_experiment(task_name, info, X, y, test_size, extras)
                    all_results.extend(res)
                    all_robustness.extend(rob)
                if "depth_curve" in extras and "alpha_curve" in extras:
                    curves = pd.concat(
                        [extras["depth_curve"], extras["alpha_curve"]], ignore_index=True
                    )
                    curves.to_csv(
                        RESULTS_DIR / f"courbes_elagage_{task_name}.csv",
                        index=False, encoding="utf-8-sig",
                    )
                    save_pruning_figure(
                        task_name, info["titre"],
                        extras["depth_curve"], extras["alpha_curve"],
                    )
            except Exception:
                failed.append(task_name)
                print(f"\nERREUR dans la tache {task_name} :")
                traceback.print_exc()
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    if not all_results:
        print("\nAucun resultat genere. Verifie les messages d'erreur ci-dessus.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "comparaison_modeles.csv",
                      index=False, encoding="utf-8-sig")
    if all_robustness:
        pd.DataFrame(all_robustness).to_csv(
            RESULTS_DIR / "robustesse_cv.csv", index=False, encoding="utf-8-sig"
        )

    print("\n" + "=" * 70)
    print("RESUME - split 80/20")
    print("=" * 70)
    resume = results_df[results_df["split"] == "80_20"][
        ["task", "model", "accuracy", "f1_macro", "recall_hit",
         "precision_hit", "ecart_f1_macro_train_test"]
    ].round(3)
    print(resume.to_string(index=False))

    print(f"\nResultats : {RESULTS_DIR / 'comparaison_modeles.csv'}")
    print(f"Figures   : {FIGURES_DIR}")
    if failed:
        print(f"ATTENTION - taches en echec : {failed}")
    print(f"Duree totale : {(time.time() - start_all) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
