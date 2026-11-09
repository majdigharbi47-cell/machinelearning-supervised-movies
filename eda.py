
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. CONFIGURATION ET CHARGEMENT
# =========================================================

DOSSIER_DATA = "data_cleaned"
DOSSIER_FIGURES = "figures"

os.makedirs(DOSSIER_FIGURES, exist_ok=True)

print("\n========== 1. CHARGEMENT DES DATASETS ==========")

fichiers = [
    "financial_movies.csv",
    "recommendation_catalog.csv",
    "audience_analysis.csv"
]

for fichier in fichiers:
    chemin = os.path.join(DOSSIER_DATA, fichier)
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"{chemin} introuvable. Lance d'abord nettoyage.py."
        )

financial_df = pd.read_csv(
    f"{DOSSIER_DATA}/financial_movies.csv",
    low_memory=False
)

recommendation_df = pd.read_csv(
    f"{DOSSIER_DATA}/recommendation_catalog.csv",
    low_memory=False
)

audience_df = pd.read_csv(
    f"{DOSSIER_DATA}/audience_analysis.csv",
    low_memory=False
)

print("Financial :", financial_df.shape)
print("Recommendation :", recommendation_df.shape)
print("Audience :", audience_df.shape)

# =========================================================
# 2. PREPARATION
# =========================================================

print("\n========== 2. PREPARATION ==========")

variables = [
    "budget", "revenue", "runtime",
    "popularity", "vote_average", "vote_count"
]

for col in variables:
    financial_df[col] = pd.to_numeric(
        financial_df[col], errors="coerce"
    )

for col in ["popularity", "vote_average", "vote_count"]:
    audience_df[col] = pd.to_numeric(
        audience_df[col], errors="coerce"
    )

recommendation_df["release_year"] = pd.to_numeric(
    recommendation_df["release_year"], errors="coerce"
)

financial_df["release_year"] = pd.to_numeric(
    financial_df["release_year"], errors="coerce"
)

financial_df["genres"] = (
    financial_df["genres"].fillna("Unknown").astype(str)
)

recommendation_df["genres"] = (
    recommendation_df["genres"].fillna("Unknown").astype(str)
)

# =========================================================
# 3. STATISTIQUES DESCRIPTIVES
# =========================================================

print("\n========== 3. STATISTIQUES DESCRIPTIVES ==========")

statistiques = financial_df[variables].describe()

print(statistiques)

statistiques.to_csv(
    f"{DOSSIER_DATA}/eda_statistiques.csv",
    encoding="utf-8-sig"
)

# =========================================================
# 4. REPARTITION HIT / FLOP
# =========================================================

print("\n========== 4. REPARTITION HIT / FLOP ==========")

classes = financial_df["financial_target"].value_counts()

print(classes)
print(
    financial_df["financial_target"].value_counts(normalize=True) * 100
)

plt.figure(figsize=(8, 5))
classes.plot(kind="bar")

plt.title("Répartition des films Hit et Flop")
plt.xlabel("Classe financière")
plt.ylabel("Nombre de films")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/01_repartition_hit_flop.png",
    dpi=300
)
plt.close()

# =========================================================
# 5. DISTRIBUTIONS NORMALES ET LOGARITHMIQUES
# =========================================================

print("\n========== 5. DISTRIBUTIONS ==========")

noms = {
    "budget": "Budget",
    "revenue": "Revenue",
    "runtime": "Runtime",
    "popularity": "Popularity",
    "vote_average": "Vote Average",
    "vote_count": "Vote Count"
}

for i, (col, titre) in enumerate(noms.items(), start=2):

    valeurs = financial_df[col].dropna()

    # Distribution normale
    plt.figure(figsize=(9, 5))
    plt.hist(valeurs, bins=50)

    plt.title(f"Distribution du {titre}")
    plt.xlabel(titre)
    plt.ylabel("Nombre de films")
    plt.tight_layout()

    plt.savefig(
        f"{DOSSIER_FIGURES}/{i:02d}_distribution_{col}.png",
        dpi=300
    )
    plt.close()

    # Distribution logarithmique
    if col in ["budget", "revenue", "popularity", "vote_count"]:

        valeurs_log = np.log1p(valeurs[valeurs >= 0])

        plt.figure(figsize=(9, 5))
        plt.hist(valeurs_log, bins=50)

        plt.title(f"Distribution du {titre} (Log Scale)")
        plt.xlabel(f"Log(1 + {titre})")
        plt.ylabel("Nombre de films")
        plt.tight_layout()

        plt.savefig(
            f"{DOSSIER_FIGURES}/{i:02d}_distribution_{col}_log.png",
            dpi=300
        )
        plt.close()

# =========================================================
# 6. BOXPLOTS
# =========================================================

print("\n========== 6. BOXPLOTS ==========")

for col, titre in noms.items():

    valeurs = financial_df[col].dropna()

    plt.figure(figsize=(9, 4))
    plt.boxplot(valeurs, vert=False)

    plt.title(f"Boxplot du {titre}")
    plt.xlabel(titre)
    plt.tight_layout()

    plt.savefig(
        f"{DOSSIER_FIGURES}/boxplot_{col}.png",
        dpi=300
    )
    plt.close()

# =========================================================
# 7. BUDGET VS REVENUE
# =========================================================

print("\n========== 7. BUDGET VS REVENUE ==========")

plt.figure(figsize=(9, 6))

plt.scatter(
    financial_df["budget"],
    financial_df["revenue"],
    alpha=0.4
)

plt.title("Relation entre Budget et Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/budget_vs_revenue.png",
    dpi=300
)
plt.close()

# Version logarithmique
plt.figure(figsize=(9, 6))

plt.scatter(
    np.log1p(financial_df["budget"]),
    np.log1p(financial_df["revenue"]),
    alpha=0.4
)

plt.title("Relation Budget / Revenue (Log Scale)")
plt.xlabel("Log(1 + Budget)")
plt.ylabel("Log(1 + Revenue)")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/budget_vs_revenue_log.png",
    dpi=300
)
plt.close()

# =========================================================
# 8. MATRICE DE CORRELATION
# =========================================================

print("\n========== 8. MATRICE DE CORRELATION ==========")

correlation = financial_df[variables].corr()

print(correlation)

correlation.to_csv(
    f"{DOSSIER_DATA}/correlation_matrix.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 8))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Coefficient de corrélation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        valeur = correlation.iloc[i, j]
        if pd.notna(valeur):
            plt.text(
                j, i, f"{valeur:.2f}",
                ha="center", va="center"
            )

plt.title("Matrice de corrélation")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/matrice_correlation.png",
    dpi=300
)
plt.close()

# =========================================================
# 9. TOP 10 GENRES
# =========================================================

print("\n========== 9. TOP 10 GENRES ==========")

genres = (
    recommendation_df["genres"]
    .str.split(",")
    .explode()
    .str.strip()
)

genres = genres[
    (genres != "") &
    (genres.str.lower() != "unknown")
]

top_genres = genres.value_counts().head(10)

print(top_genres)

plt.figure(figsize=(10, 6))

top_genres.sort_values().plot(kind="barh")

plt.title("Top 10 des genres les plus fréquents")
plt.xlabel("Nombre de films")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/top_10_genres.png",
    dpi=300
)
plt.close()

# =========================================================
# 10. TAUX DE HIT PAR GENRE
# =========================================================

print("\n========== 10. TAUX DE HIT PAR GENRE ==========")

films_genres = financial_df[
    ["id", "genres", "financial_target"]
].copy()

films_genres["genres"] = films_genres["genres"].str.split(",")
films_genres = films_genres.explode("genres")
films_genres["genres"] = films_genres["genres"].str.strip()

films_genres = films_genres[
    (films_genres["genres"] != "") &
    (films_genres["genres"].str.lower() != "unknown")
].copy()

films_genres["is_hit"] = (
    films_genres["financial_target"] == "Financial_Hit"
).astype(int)

analyse_genres = films_genres.groupby("genres").agg(
    nombre_films=("id", "count"),
    taux_hit=("is_hit", "mean")
)

analyse_genres["taux_hit"] *= 100

# Garder les genres avec au moins 30 films
analyse_genres = analyse_genres[
    analyse_genres["nombre_films"] >= 30
].sort_values("taux_hit", ascending=False)

print(analyse_genres)

analyse_genres.to_csv(
    f"{DOSSIER_DATA}/analyse_hit_par_genre.csv",
    encoding="utf-8-sig"
)

top_hit_genres = analyse_genres.head(10)

plt.figure(figsize=(10, 6))

top_hit_genres["taux_hit"].sort_values().plot(kind="barh")

plt.title("Top 10 des genres avec le meilleur taux de Hit")
plt.xlabel("Taux de Hit (%)")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/taux_hit_par_genre.png",
    dpi=300
)
plt.close()

# =========================================================
# 11. NOMBRE DE FILMS PAR ANNEE
# =========================================================

print("\n========== 11. FILMS PAR ANNEE ==========")

films_par_annee = (
    recommendation_df.dropna(subset=["release_year"])
    .groupby("release_year")["id"]
    .count()
)

plt.figure(figsize=(12, 5))

films_par_annee.plot()

plt.title("Nombre de films par année")
plt.xlabel("Année")
plt.ylabel("Nombre de films")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/films_par_annee.png",
    dpi=300
)
plt.close()

# =========================================================
# 12. HIT / FLOP PAR ANNEE
# =========================================================

print("\n========== 12. HIT / FLOP PAR ANNEE ==========")

hit_flop_annee = (
    financial_df.dropna(subset=["release_year"])
    .groupby(["release_year", "financial_target"])
    .size()
    .unstack(fill_value=0)
)

hit_flop_annee.to_csv(
    f"{DOSSIER_DATA}/hit_flop_par_annee.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 5))

hit_flop_annee.plot(ax=plt.gca())

plt.title("Evolution des Hit et Flop par année")
plt.xlabel("Année")
plt.ylabel("Nombre de films")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/hit_flop_par_annee.png",
    dpi=300
)
plt.close()

# =========================================================
# 13. TAUX DE HIT PAR ANNEE
# =========================================================

print("\n========== 13. TAUX DE HIT PAR ANNEE ==========")

taux_hit_annee = (
    financial_df.dropna(subset=["release_year"])
    .groupby("release_year")["financial_target"]
    .apply(
        lambda x: (x == "Financial_Hit").mean() * 100
    )
)

nombre_films_annee = (
    financial_df.dropna(subset=["release_year"])
    .groupby("release_year")["id"]
    .count()
)

taux_hit_annee = taux_hit_annee[
    nombre_films_annee >= 20
]

plt.figure(figsize=(12, 5))

taux_hit_annee.plot()

plt.title("Taux de Hit par année (minimum 20 films)")
plt.xlabel("Année")
plt.ylabel("Taux de Hit (%)")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/taux_hit_par_annee.png",
    dpi=300
)
plt.close()

# =========================================================
# 14. ANALYSE DU PUBLIC
# =========================================================

print("\n========== 14. ANALYSE DU PUBLIC ==========")

plt.figure(figsize=(9, 5))

plt.hist(
    audience_df["vote_average"].dropna(),
    bins=30
)

plt.title("Distribution des évaluations du public")
plt.xlabel("Vote Average")
plt.ylabel("Nombre de films")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/audience_vote_average.png",
    dpi=300
)
plt.close()

audience_valid = audience_df.dropna(
    subset=["popularity", "vote_average"]
)

plt.figure(figsize=(9, 6))

plt.scatter(
    audience_valid["vote_average"],
    audience_valid["popularity"],
    alpha=0.3
)

plt.title("Popularité selon la note moyenne")
plt.xlabel("Vote Average")
plt.ylabel("Popularity")
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/popularity_vs_vote_average.png",
    dpi=300
)
plt.close()

# =========================================================
# 15. BUDGET MOYEN PAR CLASSE
# =========================================================


# =========================================================
# 15. ANALYSE DU PUBLIC
# =========================================================

print("\n========== 15. ANALYSE DU PUBLIC ==========")

# Dataset complet pour l'analyse du public
colonnes_audience = [
    "id", "title", "release_date", "release_year",
    "genres", "popularity", "vote_average", "vote_count"
]

audience_df = recommendation_df[colonnes_audience].copy()

audience_df.to_csv(
    f"{DOSSIER_DATA}/audience_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Audience :", audience_df.shape)


# =========================================================
# 15.1 AUDIENCE SUCCESS CLASSIFICATION
# =========================================================


# =========================================================
# 15. ANALYSE DU PUBLIC
# =========================================================

print("\n========== 14. ANALYSE DU PUBLIC ==========")

# Distribution des notes du public
plt.figure(figsize=(10, 6))

sns.histplot(
    audience_df["vote_average"].dropna(),
    bins=20,
    kde=True
)

plt.title("Distribution des notes des films")
plt.xlabel("Vote Average")
plt.ylabel("Nombre de films")

plt.savefig(
    f"{DOSSIER_FIGURES}/audience_vote_average.png",
    bbox_inches="tight"
)

plt.close()


# Popularité selon la note moyenne
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=audience_df,
    x="vote_average",
    y="popularity",
    alpha=0.3
)

plt.title("Popularité selon la note moyenne")
plt.xlabel("Vote Average")
plt.ylabel("Popularity")

plt.savefig(
    f"{DOSSIER_FIGURES}/popularity_vs_vote_average.png",
    bbox_inches="tight"
)

plt.close()

print("Analyse du public terminée.")



# =========================================================
# 16. REVENUE MOYEN PAR CLASSE
# =========================================================

print("\n========== 16. REVENUE MOYEN PAR CLASSE ==========")

revenue_classe = financial_df.groupby(
    "financial_target"
)["revenue"].mean()

print(revenue_classe)

plt.figure(figsize=(8, 5))

revenue_classe.plot(kind="bar")

plt.title("Revenue moyen selon la classe financière")
plt.xlabel("Classe")
plt.ylabel("Revenue moyen")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{DOSSIER_FIGURES}/revenue_moyen_par_classe.png",
    dpi=300
)
plt.close()

# =========================================================
# 17. FIN
# =========================================================

print("\n========== RAPPORT FINAL EDA ==========")

print("Films financiers :", len(financial_df))
print("Films recommendation :", len(recommendation_df))
print("Films audience :", len(audience_df))

print("Figures enregistrées dans :", DOSSIER_FIGURES)
print("Statistiques enregistrées dans :", DOSSIER_DATA)

print("\n========== EDA TERMINEE ==========")
