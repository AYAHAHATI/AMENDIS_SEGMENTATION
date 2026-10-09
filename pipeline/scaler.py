"""
scaler.py

Standardisation des variables utilisées par les modèles.

Le scaler est :
- entraîné sur les profils janvier-avril 2022-2025 (ml/train_kmeans.py) ;
- sauvegardé dans models/scaler.pkl ;
- réutilisé tel quel pour les données 2026 (jamais réajusté sur 2026).
"""

import joblib
import pandas as pd

from config.config import INTERMEDIATE_DIR, ID_COLUMN, MODEL_FEATURES, SCALER_FILE


class DataScaler:

    def transform(self, df, scaler):
        missing = [c for c in [ID_COLUMN] + MODEL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        result = df[[ID_COLUMN]].copy()
        result[MODEL_FEATURES] = scaler.transform(df[MODEL_FEATURES])
        if "PROFIL_COMPLET" in df.columns:
            result["PROFIL_COMPLET"] = df["PROFIL_COMPLET"].values
        return result

    def run_test(self, input_file, output_file):
        """Standardise un fichier avec le scaler historique."""
        input_path = INTERMEDIATE_DIR / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")
        if not SCALER_FILE.exists():
            raise FileNotFoundError(
                f"Scaler introuvable : {SCALER_FILE} (lancer main.py)"
            )

        scaler = joblib.load(SCALER_FILE)
        print(f"Scaler utilisé : {SCALER_FILE}")
        result = self.transform(pd.read_csv(input_path), scaler)

        output_path = INTERMEDIATE_DIR / output_file
        result.to_csv(output_path, index=False)
        print(f"Fichier enregistré : {output_path} ({len(result):,} lignes)")
        return result
