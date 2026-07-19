"""
transform.py

Transformation des données du projet Amendis.
"""

import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class DataTransformer:
    """
    Classe responsable de la transformation des données.
    """

    def __init__(self):
        pass

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = BASE_DIR / "data" / "intermediate" / filename

        return pd.read_csv(file_path)

    def remove_duplicates(self, df):
        """
        Supprime les lignes dupliquées.
        """

        print("\n" + "=" * 60)
        print("SUPPRESSION DES DOUBLONS")
        print("=" * 60)

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        print(f"Lignes avant : {before}")
        print(f"Lignes après : {after}")
        print(f"Doublons supprimés : {before - after}")

        return df

    def handle_missing_values(self, df):
        """
        Traite les valeurs manquantes.
        """

        print("\n" + "=" * 60)
        print("TRAITEMENT DES VALEURS MANQUANTES")
        print("=" * 60)

        if "NOM_AGC" in df.columns:
            missing = df["NOM_AGC"].isnull().sum()
            print(f"NOM_AGC : {missing} valeurs manquantes")
            df["NOM_AGC"] = df["NOM_AGC"].fillna("Inconnu")

        if "VOL_CONSO" in df.columns:
            missing = df["VOL_CONSO"].isnull().sum()
            print(f"VOL_CONSO : {missing} valeurs manquantes")

            df["VOL_CONSO"] = pd.to_numeric(
                df["VOL_CONSO"],
                errors="coerce"
            )

            mediane = df["VOL_CONSO"].median()

            print(f"Médiane utilisée : {mediane}")

            df["VOL_CONSO"] = df["VOL_CONSO"].fillna(mediane)

        if "PUI_DIA" in df.columns:
            missing = df["PUI_DIA"].isnull().sum()
            print(f"PUI_DIA : {missing} valeurs manquantes")

            df["PUI_DIA"] = pd.to_numeric(
                df["PUI_DIA"],
                errors="coerce"
            )

            mediane = df["PUI_DIA"].median()

            print(f"Médiane utilisée : {mediane}")

            df["PUI_DIA"] = df["PUI_DIA"].fillna(mediane)

        print("\nTraitement des valeurs manquantes terminé.")

        return df

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

    def show_statistics(self, df):
        """
        Affiche les statistiques des colonnes numériques.
        """

        print("\n" + "=" * 60)
        print("STATISTIQUES DES COLONNES NUMÉRIQUES")
        print("=" * 60)

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            print("Aucune colonne numérique trouvée.")
        else:
            print(numeric_df.describe())

    def run(self, input_file, output_file):
        """
        Exécute toute la transformation.
        """

        df = self.load_dataframe(input_file)

        df = self.remove_duplicates(df)

        df = self.handle_missing_values(df)

        self.show_statistics(df)

        self.save_dataframe(df, output_file)

        return df