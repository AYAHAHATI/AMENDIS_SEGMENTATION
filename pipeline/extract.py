"""
extract.py

Extraction des données du projet Amendis.
"""

import pandas as pd
from pathlib import Path

from pipeline.history import detect_separator


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# CLASSE D'EXTRACTION
# ==========================================================

class DataExtractor:
    """
    Classe responsable de l'extraction des données.
    """

    def __init__(self, data_path="data/raw"):

        self.data_path = BASE_DIR / data_path


    # ======================================================
    # LECTURE DES FICHIERS CSV HISTORIQUES
    # ======================================================

    def read_csv(self, filename):
        """
        Lit un fichier CSV historique.
        """

        file_path = self.data_path / filename

        print("=" * 60)
        print("DEBUT DE LA LECTURE")
        print("=" * 60)

        print(f"Fichier : {file_path}")

        if not file_path.exists():

            raise FileNotFoundError(
                f"Le fichier '{filename}' est introuvable."
            )

        print("Le fichier existe.")

        encodings = [
            "utf-8",
            "cp1252",
            "latin1"
        ]

        for encoding in encodings:

            try:

                print(
                    f"\nTentative avec l'encodage : {encoding}"
                )

                # Lecture de toutes les lignes
                df = pd.read_csv(
                    file_path,
                    sep=detect_separator(file_path),
                    encoding=encoding,
                    low_memory=False
                )

                print("Lecture réussie.")

                print(
                    f"Shape : {df.shape}"
                )

                return df

            except UnicodeDecodeError:

                print(
                    f"Échec avec {encoding}"
                )

            except Exception as e:

                print("Erreur pendant read_csv")

                print(type(e))

                print(e)

        raise ValueError(
            f"Impossible de lire le fichier '{filename}'."
        )


    # ======================================================
    # LECTURE DU FICHIER 2026
    # ======================================================

    def read_txt_2026(self, filename):
        """
        Lit le fichier TXT 2026.

        Le fichier 2026 utilise :
        - une séparation par tabulation
        - un encodage compatible cp1252
        - toutes les lignes sont chargées
        """

        file_path = self.data_path / filename

        print("=" * 60)
        print("LECTURE DU FICHIER 2026")
        print("=" * 60)

        print(
            f"Fichier : {file_path}"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Le fichier '{filename}' est introuvable."
            )

        print("Le fichier existe.")

        encodings = [
            "cp1252",
            "latin1",
            "utf-8"
        ]

        for encoding in encodings:

            try:

                print(
                    f"\nTentative avec l'encodage : {encoding}"
                )

                df = pd.read_csv(
                    file_path,
                    sep="\t",
                    encoding=encoding,
                    low_memory=False
                )

                print("Lecture réussie.")

                print(
                    f"Shape : {df.shape}"
                )

                return df

            except UnicodeDecodeError:

                print(
                    f"Échec avec {encoding}"
                )

            except Exception as e:

                print(
                    "Erreur pendant read_csv"
                )

                print(type(e))

                print(e)

        raise ValueError(
            f"Impossible de lire le fichier '{filename}'."
        )


    # ======================================================
    # LECTURE DE PLUSIEURS FICHIERS
    # ======================================================

    def read_all_files(self, files):

        datasets = {}

        for file in files:

            print(
                f"\nLecture : {file}"
            )

            df = self.read_csv(file)

            self.show_info(df)

            datasets[file] = df

        return datasets


    # ======================================================
    # SAUVEGARDE
    # ======================================================

    def save_dataframe(self, df, filename):

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

        print("=" * 60)
        print("SAUVEGARDE")
        print("=" * 60)

        df.to_csv(
            file_path,
            index=False
        )

        print(
            f"Fichier sauvegardé : {file_path}"
        )


    # ======================================================
    # INFORMATIONS SUR LES DONNÉES
    # ======================================================

    def show_info(self, df):

        print("=" * 60)
        print("INFORMATIONS")
        print("=" * 60)

        print(
            f"Nombre de lignes : {len(df)}"
        )

        print(
            f"Nombre de colonnes : {len(df.columns)}"
        )

        print("\nColonnes :")

        print(
            list(df.columns)
        )

        print("\nAperçu :")

        print(
            df.head()
        )


    # ======================================================
    # PIPELINE D'EXTRACTION CSV
    # ======================================================

    def run(self, input_files, output_file):

        print("=" * 60)
        print("EXTRACTION DES DONNÉES")
        print("=" * 60)

        # --------------------------------------------------
        # Plusieurs fichiers
        # --------------------------------------------------

        if isinstance(input_files, list):

            dataframes = []

            for file in input_files:

                print(
                    f"\nLecture : {file}"
                )

                df = self.read_csv(file)

                self.show_info(df)

                dataframes.append(df)

            print(
                "\nConcaténation des fichiers..."
            )

            df_final = pd.concat(
                dataframes,
                ignore_index=True
            )

            print(
                f"Nombre total de lignes : "
                f"{len(df_final)}"
            )

        # --------------------------------------------------
        # Un seul fichier
        # --------------------------------------------------

        else:

            print(
                f"\nLecture : {input_files}"
            )

            df_final = self.read_csv(
                input_files
            )

            self.show_info(
                df_final
            )

        # --------------------------------------------------
        # Sauvegarde
        # --------------------------------------------------

        self.save_dataframe(
            df_final,
            output_file
        )

        print(
            "Extraction terminée."
        )

        return df_final


    # ======================================================
    # PIPELINE D'EXTRACTION 2026
    # ======================================================

    def run_2026(self, input_file, output_file):

        print("\n" + "=" * 60)
        print("EXTRACTION DES DONNÉES 2026")
        print("=" * 60)

        # --------------------------------------------------
        # Lecture du fichier TXT 2026
        # --------------------------------------------------

        df = self.read_txt_2026(
            input_file
        )

        # --------------------------------------------------
        # Informations
        # --------------------------------------------------

        self.show_info(
            df
        )

        # --------------------------------------------------
        # Sauvegarde
        # --------------------------------------------------

        self.save_dataframe(
            df,
            output_file
        )

        print(
            "\nExtraction 2026 terminée."
        )

        return df