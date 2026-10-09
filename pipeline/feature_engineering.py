"""
feature_engineering.py

Prépare les variables de consommation à partir des profils agrégés.

Chaque contrat garde ses 5 variables (dont NB_RELEVES) et un
indicateur PROFIL_COMPLET (au moins MIN_RELEVES relevés dans la
fenêtre). Seuls les profils complets sont ensuite segmentés.
"""

import pandas as pd

from config.config import INTERMEDIATE_DIR, ID_COLUMN, FEATURES, MIN_RELEVES


class FeatureEngineer:

    def prepare_features(self, df):
        missing = [c for c in [ID_COLUMN] + FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        features = df[[ID_COLUMN] + FEATURES].copy()
        features[FEATURES] = features[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )

        before = len(features)
        features = features.dropna(subset=FEATURES).reset_index(drop=True)
        print(f"Profils : {before:,} | après suppression des NaN : {len(features):,}")

        duplicates = features[ID_COLUMN].duplicated().sum()
        if duplicates:
            raise ValueError(f"{duplicates} identifiants contrats dupliqués.")

        features["PROFIL_COMPLET"] = features["NB_RELEVES"] >= MIN_RELEVES
        n_incomplete = int((~features["PROFIL_COMPLET"]).sum())
        print(
            f"Profils incomplets (< {MIN_RELEVES} relevés) : {n_incomplete:,} "
            f"({n_incomplete / max(len(features), 1):.2%})"
        )
        return features

    def run(self, input_file, output_file):
        input_path = INTERMEDIATE_DIR / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

        features = self.prepare_features(pd.read_csv(input_path))

        output_path = INTERMEDIATE_DIR / output_file
        features.to_csv(output_path, index=False)
        print(f"Fichier enregistré : {output_path}")
        return features
