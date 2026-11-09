# -*- coding: utf-8 -*-
"""
outils_ml.py
------------
Petit module partage par machine_learning.py et par l'application finale.

Pourquoi un fichier a part ? Quand on sauvegarde un modele avec joblib,
Python doit pouvoir retrouver la classe MultiValueEncoder pour le recharger
(dans l'application, par exemple). Une classe definie dans un module importable
se recharge sans probleme ; une classe definie dans le script principal, non.

A placer dans le dossier prevs, a cote de machine_learning.py.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MultiValueEncoder(BaseEstimator, TransformerMixin):
    """
    "Action, Adventure" -> [1, 1, 0, ...] (une colonne 0/1 par categorie).
    Les categories les plus frequentes sont apprises sur le TRAIN seulement.
    """

    def __init__(self, max_categories=30, min_frequency=20):
        self.max_categories = max_categories
        self.min_frequency = min_frequency

    @staticmethod
    def _split_values(value):
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text.lower() in {"unknown", "nan", "none", "[]"}:
            return []
        items = []
        for part in re.split(r"[,|;]", text):
            item = part.strip().strip("[]{}()'\" ")
            if item and item.lower() not in {"unknown", "nan", "none"}:
                items.append(item)
        return items

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.categories_ = {}
        for column in self.columns_:
            counts = {}
            for value in X[column]:
                for item in set(self._split_values(value)):
                    counts[item] = counts.get(item, 0) + 1
            frequent = [
                item
                for item, count in sorted(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )
                if count >= self.min_frequency
            ]
            self.categories_[column] = frequent[: self.max_categories]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        blocks = []
        for column in self.columns_:
            index = {c: i for i, c in enumerate(self.categories_[column])}
            matrix = np.zeros((len(X), len(index)), dtype=np.float32)
            for row, value in enumerate(X[column]):
                for item in set(self._split_values(value)):
                    if item in index:
                        matrix[row, index[item]] = 1.0
            blocks.append(matrix)
        return np.concatenate(blocks, axis=1)

    def get_feature_names_out(self, input_features=None):
        names = []
        for column in self.columns_:
            names.extend(f"{column}__{c}" for c in self.categories_[column])
        return np.array(names, dtype=object)
