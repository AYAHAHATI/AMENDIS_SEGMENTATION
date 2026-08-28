"""
scaler.py

Standardisation des variables de consommation
pour le Machine Learning.

Le scaler est :
- entraîné sur les données d'apprentissage (2020-2024)
- sauvegardé dans models/scaler.pkl
- réutilisé pour transformer les données de test (2026)
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import joblib


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


class DataScaler:
    """
    Standardise les variables de consommation.
    """

    # ======================================================
    # VARIABLES UTILISÉES PAR LE MACHINE LEARNING
    # ======================================================

    FEATURE_COLUMNS = [
        "CONSO_TOTALE",
        "CONSO_MOYENNE",
        "CONSO_MAX",
        "CONSO_MIN",
        "NB_RELEVES"
    ]

    ID_COLUMN = "NUM_CTA_HASH"

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
    # VÉRIFICATION
    # ======================================================

    def validate_dataframe(self, df):
        """
        Vérifie que les colonnes nécessaires
        sont présentes.
        """

        required_columns = [
            self.ID_COLUMN
        ] + self.FEATURE_COLUMNS

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Colonnes manquantes : {missing_columns}"
            )

    # ======================================================
    # ENTRAÎNEMENT DU SCALER
    # ======================================================

    def fit(self, df):
        """
        Entraîne le StandardScaler sur les données
        d'apprentissage uniquement.
        """

        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT DU STANDARD SCALER")
        print("=" * 60)

        self.validate_dataframe(df)

        features = df[self.FEATURE_COLUMNS].copy()

        print("\nVariables standardisées :")
        print(self.FEATURE_COLUMNS)

        print(
            f"\nNombre de clients : {len(features)}"
        )

        scaler = StandardScaler()

        scaler.fit(features)

        # ==================================================
        # SAUVEGARDE DU SCALER
        # ==================================================

        models_path = BASE_DIR / "models"

        models_path.mkdir(
            parents=True,
            exist_ok=True
        )

        scaler_path = models_path / "scaler.pkl"

        joblib.dump(
            scaler,
            scaler_path
        )

        print(
            f"\nStandardScaler enregistré : "
            f"{scaler_path}"
        )

        return scaler

    # ======================================================
    # TRANSFORMATION
    # ======================================================

    def transform(self, df, scaler):
        """
        Transforme les données avec un scaler
        déjà entraîné.
        """

        self.validate_dataframe(df)

        features = df[self.FEATURE_COLUMNS].copy()

        scaled_array = scaler.transform(features)

        scaled_features = pd.DataFrame(
            scaled_array,
            columns=self.FEATURE_COLUMNS
        )

        # ==================================================
        # CONSERVATION DE L'IDENTIFIANT CLIENT
        # ==================================================

        result = pd.DataFrame()

        result[self.ID_COLUMN] = (
            df[self.ID_COLUMN].values
        )

        for column in self.FEATURE_COLUMNS:

            result[column] = scaled_features[column].values

        return result

    # ======================================================
    # TRAIN
    # ======================================================

    def run_train(
        self,
        input_file,
        output_file
    ):
        """
        Entraîne le scaler sur les données d'apprentissage
        puis standardise ces mêmes données.
        """

        df = self.load_dataframe(input_file)

        scaler = self.fit(df)

        scaled_features = self.transform(
            df,
            scaler
        )

        self.save_dataframe(
            scaled_features,
            output_file
        )

        print("\nDimensions :")
        print(scaled_features.shape)

        print("\nAperçu :")
        print(scaled_features.head())

        return scaled_features

    # ======================================================
    # TEST
    # ======================================================

    def run_test(
        self,
        input_file,
        output_file
    ):
        """
        Transforme les données de test avec le scaler
        entraîné sur les données d'apprentissage.
        """

        print("\n" + "=" * 60)
        print("STANDARDISATION DES DONNÉES DE TEST")
        print("=" * 60)

        df = self.load_dataframe(input_file)

        scaler_path = (
            BASE_DIR
            / "models"
            / "scaler.pkl"
        )

        if not scaler_path.exists():

            raise FileNotFoundError(
                "Le scaler entraîné est introuvable : "
                f"{scaler_path}"
            )

        scaler = joblib.load(
            scaler_path
        )

        print(
            f"\nScaler utilisé : {scaler_path}"
        )

        scaled_features = self.transform(
            df,
            scaler
        )

        self.save_dataframe(
            scaled_features,
            output_file
        )

        print("\nDimensions :")
        print(scaled_features.shape)

        print("\nAperçu :")
        print(scaled_features.head())

        return scaled_features

    # ======================================================
    # SAUVEGARDE
    # ======================================================

    def save_dataframe(
        self,
        df,
        filename
    ):
        """
        Sauvegarde le DataFrame dans
        data/intermediate.
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