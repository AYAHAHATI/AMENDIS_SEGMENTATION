"""
feature_engineering.py

Préparation des variables de consommation
pour le Machine Learning.
"""

import pandas as pd
from pathlib import Path


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


class FeatureEngineer:
    """
    Prépare les variables de consommation
    utilisées par le modèle K-Means.
    """

    # ======================================================
    # CHARGEMENT
    # ======================================================

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = (
            BASE_DIR
            / "data"
            / "intermediate"
            / filename
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        return pd.read_csv(file_path)

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    def prepare_features(self, df):
        """
        Prépare les variables utilisées pour la segmentation.

        L'identifiant NUM_CTA_HASH est conservé afin de
        pouvoir associer chaque cluster au bon client.
        """

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        # ==================================================
        # VARIABLES NÉCESSAIRES
        # ==================================================

        required_columns = [
            "NUM_CTA_HASH",
            "CONSO_TOTALE",
            "CONSO_MOYENNE",
            "CONSO_MAX",
            "CONSO_MIN",
            "NB_RELEVES"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Colonnes manquantes : "
                f"{missing_columns}"
            )

        # ==================================================
        # SÉLECTION DES DONNÉES
        # ==================================================

        features = df[
            [
                "NUM_CTA_HASH",
                "CONSO_TOTALE",
                "CONSO_MOYENNE",
                "CONSO_MAX",
                "CONSO_MIN",
                "NB_RELEVES"
            ]
        ].copy()

        print(
            f"\nNombre de lignes avant traitement : "
            f"{len(features)}"
        )

        # ==================================================
        # VALEURS NULLLES
        # ==================================================

        print("\nValeurs nulles :")
        print(
            features[
                [
                    "CONSO_TOTALE",
                    "CONSO_MOYENNE",
                    "CONSO_MAX",
                    "CONSO_MIN",
                    "NB_RELEVES"
                ]
            ].isnull().sum()
        )

        # ==================================================
        # SUPPRESSION DES LIGNES INCOMPLÈTES
        # ==================================================

        numerical_columns = [
            "CONSO_TOTALE",
            "CONSO_MOYENNE",
            "CONSO_MAX",
            "CONSO_MIN",
            "NB_RELEVES"
        ]

        features = (
            features
            .dropna(subset=numerical_columns)
            .reset_index(drop=True)
        )

        print(
            f"\nNombre de lignes après traitement : "
            f"{len(features)}"
        )

        # ==================================================
        # VÉRIFICATION DES IDENTIFIANTS
        # ==================================================

        if features["NUM_CTA_HASH"].isnull().any():

            raise ValueError(
                "Certains clients n'ont pas de "
                "NUM_CTA_HASH."
            )

        if features["NUM_CTA_HASH"].duplicated().any():

            duplicates = (
                features["NUM_CTA_HASH"]
                .duplicated()
                .sum()
            )

            raise ValueError(
                f"{duplicates} identifiants clients "
                "sont dupliqués."
            )

        # ==================================================
        # AFFICHAGE
        # ==================================================

        print("\nVariables utilisées pour la segmentation :")

        print(
            numerical_columns
        )

        print("\nIdentifiant conservé :")
        print("NUM_CTA_HASH")

        print("\nDimensions :")
        print(features.shape)

        print("\nAperçu :")
        print(features.head())

        return features

    # ======================================================
    # SAUVEGARDE
    # ======================================================

    def save_dataframe(self, df, filename):
        """
        Sauvegarde le DataFrame dans data/intermediate.
        """

        output_path = (
            BASE_DIR
            / "data"
            / "intermediate"
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = (
            output_path
            / filename
        )

        df.to_csv(
            file_path,
            index=False
        )

        print("\n" + "=" * 60)
        print("SAUVEGARDE DES DONNÉES")
        print("=" * 60)

        print(
            f"Fichier enregistré : {file_path}"
        )

    # ======================================================
    # PIPELINE
    # ======================================================

    def run(self, input_file, output_file):
        """
        Exécute le Feature Engineering.
        """

        df = self.load_dataframe(
            input_file
        )

        features = self.prepare_features(
            df
        )

        self.save_dataframe(
            features,
            output_file
        )

        return features


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    engineer = FeatureEngineer()

    engineer.run(
        "dataset_final.csv",
        "features.csv"
    )