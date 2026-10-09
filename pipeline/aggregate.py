"""
aggregate.py

Agrégation des relevés par contrat.

La consommation est agrégée avec la même fonction que l'historique
(pipeline.history.aggregate_window), afin que les profils 2026 soient
construits exactement comme ceux utilisés pour l'entraînement.

L'agrégation de la facturation est conservée pour l'analyse
descriptive ; ses variables ne sont pas utilisées par les modèles.
"""

import pandas as pd

from config.config import INTERMEDIATE_DIR, DATE_COLUMN
from pipeline.history import aggregate_window


class DataAggregator:

    def load_dataframe(self, filename):
        file_path = INTERMEDIATE_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        if DATE_COLUMN in df.columns:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        return df

    def aggregate_consumption(self, df):
        """Profil de consommation par contrat (relevés déjà filtrés)."""
        return aggregate_window(df)

    def aggregate_billing(self, df):
        """Agrège les données de facturation par contrat."""
        df = df.copy()
        df["MNT_TTC_FAC_CLI"] = pd.to_numeric(
            df["MNT_TTC_FAC_CLI"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False),
            errors="coerce",
        )
        # Montant illisible : la facture est ignorée (pas de 0 inventé)
        df = df.dropna(subset=["MNT_TTC_FAC_CLI"])

        return (
            df.groupby("NUM_CONTRAT_HASH")
            .agg(
                MONTANT_TOTAL=("MNT_TTC_FAC_CLI", "sum"),
                MONTANT_MOYEN=("MNT_TTC_FAC_CLI", "mean"),
                MONTANT_MAX=("MNT_TTC_FAC_CLI", "max"),
                NB_FACTURES=("ID_FAC_CLI", "count"),
                VOLUME_FACTURE=("VOL_CSO_FAC_CLI", "mean"),
            )
            .reset_index()
        )

    def save_dataframe(self, df, filename):
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = INTERMEDIATE_DIR / filename
        df.to_csv(file_path, index=False)
        print(f"Fichier enregistré : {file_path} ({len(df):,} lignes)")

    def run_consumption(self, input_file, output_file):
        df = self.load_dataframe(input_file)
        profile = self.aggregate_consumption(df)
        self.save_dataframe(profile, output_file)
        return profile
