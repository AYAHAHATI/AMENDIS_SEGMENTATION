"""
aggregate.py

Agrégation des données par contrat.
"""

import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class DataAggregator:
    """
    Classe responsable de l'agrégation des données.
    """

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = BASE_DIR / "data" / "intermediate" / filename

        return pd.read_csv(file_path)

    def aggregate_consumption(self, df):
        """
        Agrège les consommations par contrat.
        """

        print("\n" + "=" * 60)
        print("AGRÉGATION DES CONSOMMATIONS")
        print("=" * 60)

        hist_client = (
            df.groupby("NUM_CTA_HASH")
            .agg(
                CONSO_TOTALE=("VOL_CONSO", "sum"),
                CONSO_MOYENNE=("VOL_CONSO", "mean"),
                CONSO_MAX=("VOL_CONSO", "max"),
                CONSO_MIN=("VOL_CONSO", "min"),
                NB_RELEVES=("VOL_CONSO", "count")
            )
            .reset_index()
        )

        print(f"\nNombre de contrats : {len(hist_client)}")

        print("\nAperçu :")
        print(hist_client.head())

        return hist_client

    def aggregate_billing(self, df):
        """
        Agrège les données de facturation par contrat.
        """

        print("\n" + "=" * 60)
        print("AGRÉGATION DES FACTURES")
        print("=" * 60)

        df["MNT_TTC_FAC_CLI"] = (
            df["MNT_TTC_FAC_CLI"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )

        df["MNT_TTC_FAC_CLI"] = pd.to_numeric(
            df["MNT_TTC_FAC_CLI"],
            errors="coerce"
        )

        df["MNT_TTC_FAC_CLI"] = df["MNT_TTC_FAC_CLI"].fillna(0)

        fact_client = (
            df.groupby("NUM_CONTRAT_HASH")
            .agg(
                MONTANT_TOTAL=("MNT_TTC_FAC_CLI", "sum"),
                MONTANT_MOYEN=("MNT_TTC_FAC_CLI", "mean"),
                MONTANT_MAX=("MNT_TTC_FAC_CLI", "max"),
                NB_FACTURES=("ID_FAC_CLI", "count"),
                VOLUME_FACTURE=("VOL_CSO_FAC_CLI", "mean")
            )
            .reset_index()
        )

        print(f"\nNombre de contrats : {len(fact_client)}")

        print("\nAperçu :")
        print(fact_client.head())

        return fact_client

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

    def run(
        self,
        hist_input,
        fact_input,
        hist_output,
        fact_output
    ):
        """
        Exécute toute l'agrégation.
        """

        hist_df = self.load_dataframe(hist_input)

        fact_df = self.load_dataframe(fact_input)

        hist_client = self.aggregate_consumption(hist_df)

        fact_client = self.aggregate_billing(fact_df)

        self.save_dataframe(hist_client, hist_output)

        self.save_dataframe(fact_client, fact_output)

        return hist_client, fact_client