"""
transform.py

Transformation des données du projet Amendis.
"""

import pandas as pd
from pathlib import Path


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# CLASSE DE TRANSFORMATION
# ==========================================================

class DataTransformer:
    """
    Classe responsable de la transformation des données.
    """

    def __init__(self):
        pass


    # ======================================================
    # CHARGEMENT DES DONNÉES
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

        print("=" * 60)
        print("CHARGEMENT DES DONNÉES")
        print("=" * 60)

        print(f"Fichier : {file_path}")

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        print(f"Nombre de lignes : {len(df)}")
        print(f"Nombre de colonnes : {len(df.columns)}")

        return df


    # ======================================================
    # SUPPRESSION DES DOUBLONS
    # ======================================================

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
        print(
            f"Doublons supprimés : {before - after}"
        )

        return df


    # ======================================================
    # TRAITEMENT DES VALEURS MANQUANTES
    # ======================================================

    def handle_missing_values(self, df):
        """
        Traite les valeurs manquantes.
        """

        print("\n" + "=" * 60)
        print("TRAITEMENT DES VALEURS MANQUANTES")
        print("=" * 60)


        # --------------------------------------------------
        # NOM_AGC
        # --------------------------------------------------

        if "NOM_AGC" in df.columns:

            missing = df["NOM_AGC"].isnull().sum()

            print(
                f"NOM_AGC : {missing} valeurs manquantes"
            )

            df["NOM_AGC"] = (
                df["NOM_AGC"]
                .fillna("Inconnu")
            )


        # --------------------------------------------------
        # VOL_CONSO
        # --------------------------------------------------

        if "VOL_CONSO" in df.columns:

            missing = df["VOL_CONSO"].isnull().sum()

            print(
                f"VOL_CONSO : {missing} valeurs manquantes"
            )

            # Conversion en numérique
            df["VOL_CONSO"] = pd.to_numeric(
                df["VOL_CONSO"],
                errors="coerce"
            )

            # Calcul de la médiane
            mediane = df["VOL_CONSO"].median()

            print(
                f"Médiane utilisée : {mediane}"
            )

            # Remplacement des valeurs manquantes
            df["VOL_CONSO"] = (
                df["VOL_CONSO"]
                .fillna(mediane)
            )


        # --------------------------------------------------
        # PUI_DIA
        # --------------------------------------------------

        if "PUI_DIA" in df.columns:

            missing = df["PUI_DIA"].isnull().sum()

            print(
                f"PUI_DIA : {missing} valeurs manquantes"
            )

            # Conversion en numérique
            df["PUI_DIA"] = pd.to_numeric(
                df["PUI_DIA"],
                errors="coerce"
            )

            # Calcul de la médiane
            mediane = df["PUI_DIA"].median()

            print(
                f"Médiane utilisée pour PUI_DIA : "
                f"{mediane}"
            )

            # Remplacement des valeurs manquantes
            df["PUI_DIA"] = (
                df["PUI_DIA"]
                .fillna(mediane)
            )


        print(
            "\nTraitement des valeurs manquantes terminé."
        )

        return df


    # ======================================================
    # STATISTIQUES
    # ======================================================

    def show_statistics(self, df):
        """
        Affiche les statistiques des colonnes numériques.
        """

        print("\n" + "=" * 60)
        print("STATISTIQUES DES COLONNES NUMÉRIQUES")
        print("=" * 60)

        numeric_df = df.select_dtypes(
            include="number"
        )

        if numeric_df.empty:

            print(
                "Aucune colonne numérique trouvée."
            )

        else:

            print(
                numeric_df.describe()
            )


    # ======================================================
    # SAUVEGARDE
    # ======================================================

    def save_dataframe(self, df, filename):
        """
        Sauvegarde un DataFrame dans data/intermediate.
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

        file_path = output_path / filename

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
    # PIPELINE COMPLET DE TRANSFORMATION
    # ======================================================

    def run(self, input_file, output_file):
        """
        Exécute toute la transformation.
        """

        print("\n" + "=" * 60)
        print("TRANSFORMATION DES DONNÉES")
        print("=" * 60)


        # --------------------------------------------------
        # 1. Chargement
        # --------------------------------------------------

        df = self.load_dataframe(
            input_file
        )


        # --------------------------------------------------
        # 2. Suppression des doublons
        # --------------------------------------------------

        df = self.remove_duplicates(
            df
        )


        # --------------------------------------------------
        # 3. Traitement des valeurs manquantes
        # --------------------------------------------------

        df = self.handle_missing_values(
            df
        )


        # --------------------------------------------------
        # 4. Statistiques
        # --------------------------------------------------

        self.show_statistics(
            df
        )


        # --------------------------------------------------
        # 5. Sauvegarde
        # --------------------------------------------------

        self.save_dataframe(
            df,
            output_file
        )


        print("\nTransformation terminée.")

        return df