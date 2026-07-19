"""
feature_engineering.py

Préparation des variables pour le Machine Learning.
"""

import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class FeatureEngineer:
    """
    Prépare les variables qui seront utilisées
    par le modèle de Machine Learning.
    """

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = BASE_DIR / "data" / "intermediate" / filename

        return pd.read_csv(file_path)

    def prepare_features(self, df):
        """
        Sélectionne les variables qui seront utilisées
        par le modèle de Machine Learning.
        """

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        features = df[
            [
                "CONSO_TOTALE",
                "CONSO_MOYENNE",
                "CONSO_MAX",
                "CONSO_MIN",
                "NB_RELEVES",
                "MONTANT_TOTAL",
                "MONTANT_MOYEN",
                "MONTANT_MAX",
                "NB_FACTURES",
                "VOLUME_FACTURE"
            ]
        ].copy()

        print("\nVariables sélectionnées :")
        print(features.columns.tolist())

        print("\nDimensions :")
        print(features.shape)

        print("\nAperçu :")
        print(features.head())

        return features

    def save_dataframe(self, df, filename):
        """
        Sauvegarde un DataFrame dans data/intermediate.
        """

        output_path = BASE_DIR / "data" / "intermediate"

        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename

        df.to_csv(file_path, index=False)

        print("\n" + "=" * 60)
        print("SAUVEGARDE DES DONNÉES")
        print("=" * 60)
        print(f"Fichier enregistré : {file_path}")

    def run(self, input_file, output_file):
        """
        Exécute le Feature Engineering.
        """

        df = self.load_dataframe(input_file)

        features = self.prepare_features(df)

        self.save_dataframe(features, output_file)

        return features