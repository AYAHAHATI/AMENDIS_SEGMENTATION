"""
scaler.py

Standardisation des variables.
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class DataScaler:
    """
    Standardise les variables numériques.
    """

    def load_dataframe(self, filename):
        """
        Charge un DataFrame depuis data/intermediate.
        """

        file_path = BASE_DIR / "data" / "intermediate" / filename

        return pd.read_csv(file_path)

    def scale(self, features):
        """
        Standardise les variables numériques.
        """

        print("\n" + "=" * 60)
        print("STANDARDISATION DES DONNÉES")
        print("=" * 60)

        scaler = StandardScaler()

        scaled_array = scaler.fit_transform(features)

        scaled_df = pd.DataFrame(
            scaled_array,
            columns=features.columns
        )

        print("\nDimensions :")
        print(scaled_df.shape)

        print("\nAperçu :")
        print(scaled_df.head())

        return scaled_df

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
        Exécute toute la standardisation.
        """

        features = self.load_dataframe(input_file)

        scaled_features = self.scale(features)

        self.save_dataframe(
            scaled_features,
            output_file
        )

        return scaled_features