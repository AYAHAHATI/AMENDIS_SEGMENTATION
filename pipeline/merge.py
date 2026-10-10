"""
merge.py

Fusion des données du projet Amendis.
"""

import pandas as pd
from pathlib import Path

from config.config import INTERMEDIATE_DIR


class DataMerger:
    """
    Classe responsable de la fusion des données.
    """

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = INTERMEDIATE_DIR / filename

        return pd.read_csv(file_path)

    def merge_data(self, hist_client, fact_client):
        """
        Fusionne les données de consommation et de facturation.
        """

        print("\n" + "=" * 60)
        print("FUSION DES DONNÉES")
        print("=" * 60)

        dataset_final = pd.merge(
            hist_client,
            fact_client,
            left_on="NUM_CTA_HASH",
            right_on="NUM_CONTRAT_HASH",
            how="left"
        )

        print(f"Nombre de lignes : {len(dataset_final)}")
        print(f"Nombre de colonnes : {dataset_final.shape[1]}")

        print("\nAperçu :")
        print(dataset_final.head())

        return dataset_final

    def save_dataframe(self, df, filename):
        """
        Sauvegarde un DataFrame dans data/intermediate.
        """

        output_path = INTERMEDIATE_DIR

        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename

        df.to_csv(file_path, index=False)

        print("\n" + "=" * 60)
        print("SAUVEGARDE DES DONNÉES")
        print("=" * 60)
        print(f"Fichier enregistré : {file_path}")

    def run(
        self,
        hist_input,
        fact_input,
        output_file
    ):
        """
        Exécute toute la fusion.
        """

        hist_client = self.load_dataframe(hist_input)

        fact_client = self.load_dataframe(fact_input)

        dataset_final = self.merge_data(
            hist_client,
            fact_client
        )

        self.save_dataframe(
            dataset_final,
            output_file
        )

        return dataset_final