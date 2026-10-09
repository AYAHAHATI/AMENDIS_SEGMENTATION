"""
predict_kmeans.py

Applique le K-Means historique (sans réentraînement) aux profils
janvier-avril 2026 déjà standardisés.

Les profils incomplets reçoivent CLUSTER = -1 ("Profil incomplet").
"""

import joblib
import pandas as pd

from config.config import (
    INTERMEDIATE_DIR,
    ID_COLUMN,
    MODEL_FEATURES,
    KMEANS_FILE,
    INCOMPLETE_CLUSTER,
)
from ml.labels import load_labels


def predict_kmeans_2026(
    input_file="scaled_features_2026.csv",
    output_file="clients_cluster_2026.csv",
):
    input_path = INTERMEDIATE_DIR / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")
    if not KMEANS_FILE.exists():
        raise FileNotFoundError(f"Modèle introuvable : {KMEANS_FILE}")

    df = pd.read_csv(input_path)
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Identifiants dupliqués dans les profils 2026.")

    model = joblib.load(KMEANS_FILE)
    labels = load_labels()

    result = df[[ID_COLUMN]].copy()
    result["CLUSTER"] = model.predict(df[MODEL_FEATURES].to_numpy())

    if "PROFIL_COMPLET" in df.columns:
        result.loc[~df["PROFIL_COMPLET"].astype(bool), "CLUSTER"] = INCOMPLETE_CLUSTER

    result["LIBELLE_CLUSTER"] = result["CLUSTER"].map(labels)

    print(f"Modèle : {KMEANS_FILE.name} | contrats : {len(result):,}")
    print(result["LIBELLE_CLUSTER"].value_counts().to_string())

    output_path = INTERMEDIATE_DIR / output_file
    result.to_csv(output_path, index=False)
    print(f"Fichier enregistré : {output_path}")
    return result


if __name__ == "__main__":
    predict_kmeans_2026()
