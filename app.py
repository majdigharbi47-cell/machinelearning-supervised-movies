# -*- coding: utf-8 -*-
"""
app.py - Application TMDB : prediction Hit / Flop et recommandation de films.

Lancer (depuis le dossier prevs, apres avoir execute
entrainer_modeles_finaux.py et construire_recommandation.py) :
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from outils_ml import MultiValueEncoder  # noqa: F401  (necessaire pour joblib)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
POSTER = "https://image.tmdb.org/t/p/w185"

st.set_page_config(page_title="TMDB : Hit ou Flop ?", page_icon="🎬", layout="wide")


# ------------------------------------------------------------
# Chargement (mis en cache : fait une seule fois)
# ------------------------------------------------------------
@st.cache_resource
def charger_modele(nom):
    return joblib.load(MODELS / f"{nom}.joblib")


@st.cache_resource
def charger_reco():
    chemin = MODELS / "reco.joblib"
    return joblib.load(chemin) if chemin.exists() else None


@st.cache_data
def charger_meta():
    return json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))


def modeles_prets():
    return (MODELS / "meta.json").exists()


def probabilite_hit(nom, ligne):
    """Probabilite de Hit = part des 200 arbres qui votent Hit."""
    modele = charger_modele(nom)
    X = pd.DataFrame([ligne])
    classes = list(modele.classes_)
    return float(modele.predict_proba(X)[0][classes.index("Hit")])


def afficher_resultat(p, nom_tache, vrai_hit="Hit", vrai_flop="Flop"):
    meta = charger_meta()["taches"][nom_tache]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilité de Hit", f"{p * 100:.0f} %")
        if p >= 0.5:
            st.success(f"Le modèle prédit : **{vrai_hit}**")
        else:
            st.error(f"Le modèle prédit : **{vrai_flop}**")
    with col2:
        st.progress(min(max(p, 0.0), 1.0))
        t = meta["test"]
        st.caption(
            f"Fiabilité du modèle sur des films jamais vus : F1-macro "
            f"{t['f1_macro']:.2f}, exactitude {t['accuracy']:.2f}. "
            f"Dans le jeu de données, {meta['part_hit'] * 100:.0f} % des films "
            f"sont des Hit."
        )
        st.warning(
            "C'est une **estimation** apprise sur des films passés, pas une "
            "garantie. Plus la probabilité est proche de 50 %, moins le modèle "
            "est sûr."
        )


def liste_vers_texte(valeurs):
    return ", ".join(valeurs) if valeurs else "Unknown"


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_accueil():
    st.title("🎬 Hit ou Flop ? Prédire le succès d'un film")
    st.write(
        "Cette application utilise le jeu de données TMDB (films) pour répondre "
        "à trois questions et recommander des films proches."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Avant la sortie**\n\nLe film rapportera-t-il au moins 2 fois son budget ?")
    c2.info("**2. Après la sortie**\n\nMême question, avec la popularité et les votes du public.")
    c3.info("**3. Succès auprès du public**\n\nLa note moyenne sera-t-elle d'au moins 7/10 ?")
    st.write(
        "**Recommandation** : choisis un film que tu aimes, l'application "
        "propose les films qui lui ressemblent (genres, mots-clés, résumé)."
    )
    if not modeles_prets():
        st.error(
            "Les modèles ne sont pas encore prêts. Lance d'abord "
            "`python entrainer_modeles_finaux.py` puis relance l'application."
        )
        return
    meta = charger_meta()["taches"]
    st.subheader("Qualité des modèles (test sur des films jamais vus)")
    tableau = pd.DataFrame([
        {
            "Tâche": m["titre"], "Films": m["n_films"],
            "Part de Hit": f"{m['part_hit'] * 100:.0f} %",
            "F1-macro": round(m["test"]["f1_macro"], 3),
            "Exactitude": round(m["test"]["accuracy"], 3),
        }
        for m in meta.values()
    ])
    st.dataframe(tableau, hide_index=True)


def formulaire_commun(choix, avec_public):
    c1, c2 = st.columns(2)
    with c1:
        budget = st.number_input("Budget (en $)", min_value=10_000,
                                 max_value=1_000_000_000, value=20_000_000,
                                 step=1_000_000)
        annee = st.slider("Année de sortie", choix["annee_min"],
                          choix["annee_max"] + 2, min(2025, choix["annee_max"]))
        mois = st.selectbox("Mois de sortie", list(range(1, 13)), index=5)
        langue = st.selectbox("Langue originale", choix["langues"])
    with c2:
        genres = st.multiselect("Genres", choix["genres"], default=choix["genres"][:1])
        pays = st.multiselect("Pays de production", choix["pays"])
        societes = st.multiselect("Sociétés de production", choix["societes"])
    ligne = {
        "budget": float(budget), "release_year": annee, "release_month": mois,
        "original_language": langue,
        "genres": liste_vers_texte(genres),
        "production_companies": liste_vers_texte(societes),
        "production_countries": liste_vers_texte(pays),
    }
    return ligne


def page_financier(apres):
    nom = "financial_post" if apres else "financial_pre"
    st.title("💰 " + ("Succès financier après la sortie" if apres
                       else "Succès financier avant la sortie"))
    st.write("Hit = le film rapporte **au moins 2 fois son budget**.")
    if not modeles_prets():
        st.error("Modèles absents : lance `entrainer_modeles_finaux.py`.")
        return
    choix = charger_meta()["choix"]
    ligne = formulaire_commun(choix, apres)
    if apres:
        st.subheader("Signaux du public (disponibles après la sortie)")
        c1, c2 = st.columns(2)
        with c1:
            ligne["runtime"] = float(st.number_input("Durée (minutes)", 40, 300, 110))
            ligne["vote_average"] = float(st.slider("Note moyenne", 1.0, 10.0, 6.5, 0.1))
        with c2:
            ligne["popularity"] = float(st.number_input("Popularité TMDB", 0.0, 5000.0, 20.0))
            ligne["vote_count"] = float(st.number_input("Nombre de votes", 0, 50_000, 1_500))
    if st.button("Prédire", type="primary"):
        afficher_resultat(probabilite_hit(nom, ligne), nom)
        if apres:
            st.info(
                "Après la sortie, le modèle **estime** le succès à partir de "
                "signaux liés aux recettes (le nombre de votes surtout) : ce "
                "n'est pas une prévision."
            )


def page_public():
    st.title("⭐ Succès auprès du public")
    st.write("Hit = note moyenne d'**au moins 7/10** (films avec au moins 100 votes).")
    if not modeles_prets():
        st.error("Modèles absents : lance `entrainer_modeles_finaux.py`.")
        return
    choix = charger_meta()["choix"]
    c1, c2 = st.columns(2)
    with c1:
        annee = st.slider("Année de sortie", choix["annee_min"],
                          choix["annee_max"] + 2, min(2025, choix["annee_max"]))
        mois = st.selectbox("Mois de sortie", list(range(1, 13)), index=5)
        langue = st.selectbox("Langue originale", choix["langues"])
        duree = st.number_input("Durée (minutes)", 40, 300, 110)
    with c2:
        genres = st.multiselect("Genres", choix["genres"], default=choix["genres"][:1])
        pays = st.multiselect("Pays de production", choix["pays"])
        societes = st.multiselect("Sociétés de production", choix["societes"])
        mots = st.text_input("Mots-clés (séparés par des virgules, en anglais)", "")
    c3, c4 = st.columns(2)
    popularite = c3.number_input("Popularité TMDB", 0.0, 5000.0, 20.0)
    votes = c4.number_input("Nombre de votes", 100, 50_000, 1_000)
    ligne = {
        "release_year": annee, "release_month": mois, "original_language": langue,
        "genres": liste_vers_texte(genres),
        "production_companies": liste_vers_texte(societes),
        "production_countries": liste_vers_texte(pays),
        "keywords": mots.strip() or "Unknown",
        "runtime": float(duree), "popularity": float(popularite),
        "vote_count": float(votes),
    }
    if st.button("Prédire", type="primary"):
        afficher_resultat(probabilite_hit("audience", ligne), "audience",
                          "Audience Hit", "Audience Flop")


def page_reco():
    st.title("🍿 Recommandation de films")
    reco = charger_reco()
    if reco is None:
        st.error("Lance d'abord `python construire_recommandation.py`.")
        return
    films, tfidf, matrice = reco["films"], reco["tfidf"], reco["matrice"]
    st.write(
        "Le système compare le **contenu** des films (genres, mots-clés, "
        "résumé). Il ne regarde pas ce que les autres utilisateurs ont aimé."
    )
    mode = st.radio("Je cherche à partir de :", ["Un film que j'aime",
                                                  "Une description libre"],
                    horizontal=True)
    n = st.slider("Nombre de recommandations", 3, 15, 6)
    scores, exclure = None, None

    if mode == "Un film que j'aime":
        recherche = st.text_input("Début du titre", "")
        trouves = films[films["title"].str.contains(recherche, case=False,
                                                    na=False, regex=False)]
        trouves = trouves.nlargest(30, "vote_count") if recherche else films.head(0)
        if len(trouves):
            etiquettes = {
                f"{r.title} ({int(r.release_year) if pd.notna(r.release_year) else '?'})": i
                for i, r in trouves.iterrows()
            }
            choisi = st.selectbox("Film", list(etiquettes))
            i = etiquettes[choisi]
            scores = (matrice @ matrice[i].T).toarray().ravel()
            exclure = i
        elif recherche:
            st.info("Aucun film trouvé avec ce titre.")
    else:
        texte = st.text_area("Décris le film que tu voudrais voir (en anglais)",
                             "space adventure with robots and a journey across galaxies")
        if texte.strip():
            vecteur = tfidf.transform([texte])
            scores = (matrice @ vecteur.T).toarray().ravel()

    if scores is not None:
        if exclure is not None:
            scores[exclure] = -1
        top = np.argsort(-scores)[:n]
        for rang, j in enumerate(top, start=1):
            f = films.iloc[j]
            c1, c2 = st.columns([1, 5])
            with c1:
                if isinstance(f["poster_path"], str) and f["poster_path"]:
                    st.image(POSTER + f["poster_path"], width=110)
            with c2:
                annee = int(f["release_year"]) if pd.notna(f["release_year"]) else "?"
                st.markdown(f"**{rang}. {f['title']}** ({annee})")
                st.caption(f"{f['genres']} | note {f['vote_average']:.1f}/10 | "
                           f"similarité {scores[j]:.2f}")
                st.write(f["overview"])


def page_apropos():
    st.title("ℹ️ À propos")
    st.markdown(
        """
- **Données** : TMDB, films sortis, sans contenu adulte, nettoyés (budget et
  recettes d'au moins 10 000 $ pour la tâche financière).
- **Modèle** : Random Forest de 200 arbres, choisi parce qu'il est le meilleur
  dans les 3 tâches sur 10 tirages aléatoires différents.
- **Probabilité** : la part des 200 arbres qui votent « Hit ».
- **Limites** : le modèle apprend sur des films qui ont un budget et des
  recettes connus, souvent des films plus importants que la moyenne. Après la
  sortie, il estime le succès à partir de signaux du public ; il ne prévoit pas
  l'avenir.
        """
    )
    evaluation = MODELS / "reco_eval.json"
    if evaluation.exists():
        e = json.loads(evaluation.read_text(encoding="utf-8"))
        st.subheader("Contrôle de la recommandation")
        st.write(
            f"Sur {e['films_dans_le_modele']} films, "
            f"{e['part_voisins_avec_un_genre_commun'] * 100:.0f} % des films "
            f"recommandés partagent au moins un genre avec le film de départ. "
            f"Ressemblance de genres (Jaccard) : "
            f"{e['jaccard_genres_recommandations']:.2f} pour nos recommandations "
            f"contre {e['jaccard_genres_au_hasard']:.2f} au hasard."
        )


PAGES = {
    "Accueil": page_accueil,
    "Avant la sortie": lambda: page_financier(False),
    "Après la sortie": lambda: page_financier(True),
    "Succès public": page_public,
    "Recommandation": page_reco,
    "À propos": page_apropos,
}
choix_page = st.sidebar.radio("Navigation", list(PAGES))
PAGES[choix_page]()
