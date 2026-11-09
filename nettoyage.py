
import os
import pandas as pd
import numpy as np

# =========================================================
# 1. CONFIGURATION
# =========================================================

FICHIER_SOURCE = "TMDB_movie_dataset_v11.csv"
DOSSIER_DATA = "data_cleaned"

os.makedirs(DOSSIER_DATA, exist_ok=True)

print("\n========== 1. CHARGEMENT ==========")

df = pd.read_csv(FICHIER_SOURCE, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

print("Dimensions initiales :", df.shape)

# Vérification des colonnes nécessaires
colonnes_requises = [
    "id", "title", "status", "release_date", "revenue",
    "runtime", "adult", "budget", "original_language",
    "popularity", "genres", "production_companies",
    "production_countries", "spoken_languages",
    "keywords", "vote_average", "vote_count"
]

absentes = [c for c in colonnes_requises if c not in df.columns]

if absentes:
    raise ValueError(f"Colonnes absentes : {absentes}")

# =========================================================
# 2. NETTOYAGE DES TEXTES ET DES NOMBRES
# =========================================================

print("\n========== 2. NETTOYAGE INITIAL ==========")

colonnes_texte = [
    "title", "status", "genres", "original_language",
    "production_companies", "production_countries",
    "overview", "keywords", "spoken_languages"
]

for col in colonnes_texte:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

colonnes_numeriques = [
    "vote_average", "vote_count", "revenue",
    "runtime", "budget", "popularity"
]

for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("\nVote Average avant nettoyage :")
print("Valeurs manquantes :", df["vote_average"].isna().sum())
print("Valeurs égales à 0 :", (df["vote_average"] == 0).sum())
print("Valeurs supérieures à 10 :", (df["vote_average"] > 10).sum())

print("\nExemples :")
print(df["vote_average"].value_counts().head(10))
# =========================================================
# 3. SUPPRESSION DES FILMS ADULTES
# =========================================================

print("\n========== 3. FILTRAGE ADULTE ==========")

adult_flag = (
    df["adult"].astype("string")
    .str.strip().str.lower()
    .isin(["true", "1", "yes"])
)

adult_genre = df["genres"].astype("string").str.contains(
    r"(?:^|,\s*)Adult(?:\s*,|$)",
    case=False,
    na=False,
    regex=True
)

adult_mask = adult_flag | adult_genre

print("Films adultes détectés :", int(adult_mask.sum()))

df = df.loc[~adult_mask].copy()

# Suppression de toute la colonne Adult
df.drop(columns=["adult"], inplace=True)

print("Dimensions après suppression :", df.shape)

# =========================================================
# 4. DOUBLONS
# =========================================================

print("\n========== 4. DOUBLONS ==========")

print("Doublons exacts :", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print("Doublons ID :", df.duplicated(subset=["id"]).sum())
df.drop_duplicates(subset=["id"], inplace=True)

print("Dimensions :", df.shape)

# =========================================================
# 5. TITRES
# =========================================================

print("\n========== 5. TITRES ==========")

df.dropna(subset=["title"], inplace=True)
df = df[df["title"].str.strip().ne("")].copy()

print("Dimensions :", df.shape)

# =========================================================
# 6. DATES
# =========================================================

print("\n========== 6. DATES ==========")

df["release_date"] = pd.to_datetime(
    df["release_date"], errors="coerce"
)

annee = df["release_date"].dt.year

date_ancienne = annee < 1888
date_future = df["release_date"] > pd.Timestamp.today()

print("Dates très anciennes :", int(date_ancienne.sum()))
print("Dates futures :", int(date_future.sum()))

df.loc[date_ancienne | date_future, "release_date"] = pd.NaT

df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month

# =========================================================
# 7. VALEURS NUMERIQUES
# =========================================================

print("\n========== 7. VALEURS NUMERIQUES ==========")

for col in colonnes_numeriques:
    df.loc[df[col] < 0, col] = np.nan

for col in ["budget", "revenue", "runtime"]:
    df.loc[df[col] == 0, col] = np.nan

# Runtime supérieur à 600 : valeur suspecte
df.loc[df["runtime"] > 600, "runtime"] = np.nan

# Vote Average doit être entre 0 et 10
df.loc[df["vote_average"] > 10, "vote_average"] = np.nan
# Les films sans évaluation réelle
# On considère vote_average = 0 comme une valeur manquante

df.loc[
    df["vote_average"] == 0,
    "vote_average"
] = np.nan



# Une note sans votes est considérée comme manquante
df.loc[df["vote_count"] == 0, "vote_average"] = np.nan


# =========================================================
# 8. EXPORT DES FILMS SUSPECTS
# =========================================================

print("\n========== 8. FILMS SUSPECTS ==========")

runtime_suspect = df[df["runtime"] > 300].copy()

runtime_suspect.to_csv(
    f"{DOSSIER_DATA}/films_runtime_suspect.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Films avec runtime > 300 :", len(runtime_suspect))

# =========================================================
# 9. VALEURS MANQUANTES
# =========================================================

print("\n========== 9. VALEURS MANQUANTES ==========")

colonnes_a_remplir = [
    "genres", "original_language",
    "production_companies", "production_countries",
    "spoken_languages", "keywords"
]

for col in colonnes_a_remplir:
    df[col] = df[col].fillna("Unknown")
    df[col] = df[col].replace("", "Unknown")

print(df.isnull().sum())

# =========================================================
# 10. DATASET COMPLET NETTOYE
# =========================================================

print("\n========== 10. DATASET NETTOYE ==========")

df.to_csv(
    f"{DOSSIER_DATA}/tmdb_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset nettoyé :", df.shape)

# =========================================================
# 11. DATASET RECOMMENDATION
# =========================================================

print("\n========== 11. RECOMMENDATION ==========")

recommendation_df = df[
    df["status"].str.lower() == "released"
].copy()

colonnes_recommendation = [
    "id", "title", "release_date", "release_year",
    "genres", "original_language",
    "production_companies", "production_countries",
    "overview", "keywords", "popularity",
    "vote_average", "vote_count", "runtime",
    "poster_path"
]

recommendation_df = recommendation_df[
    colonnes_recommendation
].copy()

recommendation_df.to_csv(
    f"{DOSSIER_DATA}/recommendation_catalog.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Recommendation :", recommendation_df.shape)

# =========================================================
# 12. DATASET FINANCIER
# =========================================================

print("\n========== 12. CLASSIFICATION FINANCIERE ==========")

# Seuil minimal (en dollars) pour budget ET revenue.
# Les valeurs plus petites (ex : budget = 1 $) sont presque toujours des
# erreurs de saisie et fabriquent de faux Hit / Flop.
# Mesure sur les donnees reelles : 10 000 $ retire 3 918 films sur 14 139.
SEUIL_MIN = 10_000

financial_df = df[
    (df["status"].str.lower() == "released") &
    (df["budget"] >= SEUIL_MIN) &
    (df["revenue"] >= SEUIL_MIN) &
    (df["release_date"].notna())
].copy()

print(f"Seuil minimal budget/revenue : {SEUIL_MIN} $")

financial_df["revenue_budget_ratio"] = (
    financial_df["revenue"] / financial_df["budget"]
)

financial_df["financial_target"] = np.where(
    financial_df["revenue"] >= 2 * financial_df["budget"],
    "Financial_Hit",
    "Financial_Flop"
)

# IMPORTANT : sauvegarde du fichier financier
financial_df.to_csv(
    f"{DOSSIER_DATA}/financial_movies.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset financier :", financial_df.shape)
print(financial_df["financial_target"].value_counts())
print(
    financial_df["financial_target"]
    .value_counts(normalize=True) * 100
)

# =========================================================
# 13. DATASET PRE-RELEASE
# =========================================================

print("\n========== 13. PRE-RELEASE ==========")

colonnes_pre = [
    "id", "title", "budget", "release_year",
    "release_month", "original_language", "genres",
    "production_companies", "production_countries",
    "financial_target"
]

pre_release_df = financial_df[colonnes_pre].copy()

pre_release_df.to_csv(
    f"{DOSSIER_DATA}/pre_release.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Pre-release :", pre_release_df.shape)

# =========================================================
# 14. DATASET POST-RELEASE
# =========================================================

print("\n========== 14. POST-RELEASE ==========")

colonnes_post = colonnes_pre + [
    "runtime", "popularity", "vote_average", "vote_count"
]

post_release_df = financial_df[colonnes_post].copy()

post_release_df.to_csv(
    f"{DOSSIER_DATA}/post_release.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Post-release :", post_release_df.shape)

# =========================================================
# 15. ANALYSE DU PUBLIC
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

print("\n========== AUDIENCE SUCCESS ==========")

# Garder les films avec au moins 100 votes
# et une note disponible

audience_success_df = recommendation_df[
    (recommendation_df["vote_count"] >= 100) &
    (recommendation_df["vote_average"].notna())
].copy()

# Création du Target

audience_success_df["audience_target"] = np.where(
    audience_success_df["vote_average"] >= 7,
    "Audience_Hit",
    "Audience_Flop"
)

# Sauvegarde du dataset

audience_success_df.to_csv(
    f"{DOSSIER_DATA}/audience_success.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Audience Success :", audience_success_df.shape)
print(audience_success_df["audience_target"].value_counts())

print(
    audience_success_df["audience_target"]
    .value_counts(normalize=True) * 100
)

# =========================================================
# 16. STATISTIQUES DESCRIPTIVES
# =========================================================

print("\n========== 16. STATISTIQUES ==========")

colonnes_stats = [
    "budget", "revenue", "runtime",
    "popularity", "vote_average", "vote_count"
]

statistiques = financial_df[colonnes_stats].describe()

print(statistiques)

statistiques.to_csv(
    f"{DOSSIER_DATA}/statistiques_descriptives.csv",
    encoding="utf-8-sig"
)

# =========================================================
# 17. RAPPORT FINAL
# =========================================================

print("\n========== RAPPORT FINAL ==========")

print("Dataset nettoyé :", df.shape)
print("Recommendation :", recommendation_df.shape)
print("Financier :", financial_df.shape)
print("Pre-release :", pre_release_df.shape)
print("Post-release :", post_release_df.shape)
print("Audience :", audience_df.shape)

print("\nCSV enregistrés dans :", DOSSIER_DATA)
print("\n========== NETTOYAGE TERMINE ==========")
