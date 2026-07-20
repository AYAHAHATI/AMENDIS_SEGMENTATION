"""
extract.py

Extraction des données du projet Amendis.
"""

import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class DataExtractor:
    """
    Classe responsable de l'extraction des données.
    """

    def __init__(self, data_path="data/raw"):
        self.data_path = BASE_DIR / data_path

    def read_csv(self, filename):
        """
        Lit un fichier CSV.
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

        encodings = ["utf-8", "cp1252", "latin1"]

        for encoding in encodings:

            try:

                print(f"\nTentative avec l'encodage : {encoding}")

                df = pd.read_csv(
                    file_path,
                    sep=",",
                    encoding=encoding,
                    low_memory=False,
                    nrows=500000
                )

                print("Lecture réussie.")
                print(f"Shape : {df.shape}")

                return df

            except UnicodeDecodeError:

                print(f"Échec avec {encoding}")

            except Exception as e:

                print("Erreur pendant read_csv")
                print(type(e))
                print(e)

        raise ValueError(
            f"Impossible de lire le fichier '{filename}'."
        )

    def read_all_files(self, files):

        datasets = {}

        for file in files:

            print(f"\nLecture : {file}")

            df = self.read_csv(file)

            self.show_info(df)

            datasets[file] = df

        return datasets

    def save_dataframe(self, df, filename):

        output_path = BASE_DIR / "data" / "intermediate"

        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename

        print("=" * 60)
        print("SAUVEGARDE")
        print("=" * 60)

        df.to_csv(file_path, index=False)

        print(f"Fichier sauvegardé : {file_path}")

    def show_info(self, df):

        print("=" * 60)
        print("INFORMATIONS")
        print("=" * 60)

        print(f"Nombre de lignes : {len(df)}")
        print(f"Nombre de colonnes : {len(df.columns)}")

        print("\nColonnes :")
        print(list(df.columns))

        print("\nAperçu :")
        print(df.head())

    def run(self, input_files, output_file):

        print("=" * 60)
        print("EXTRACTION DES DONNÉES")
        print("=" * 60)

        # Si plusieurs fichiers sont fournis
        if isinstance(input_files, list):

            dataframes = []

            for file in input_files:

                print(f"\nLecture : {file}")

                df = self.read_csv(file)

                self.show_info(df)

                dataframes.append(df)

            print("\nConcaténation des fichiers...")

            df_final = pd.concat(dataframes, ignore_index=True)

            print(f"Nombre total de lignes : {len(df_final)}")

        else:

            print(f"\nLecture : {input_files}")

            df_final = self.read_csv(input_files)

            self.show_info(df_final)

        self.save_dataframe(df_final, output_file)

        print("Extraction terminée.")

        return df_final