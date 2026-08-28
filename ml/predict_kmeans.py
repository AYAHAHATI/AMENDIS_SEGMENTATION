"""
predict_kmeans.py

Application du modèle K-Means historique
sur les données de test 2026.
"""

import pandas as pd
import joblib
from pathlib import Path


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# VARIABLES UTILISÉES PAR LE MODÈLE
# ==========================================================

FEATURE_COLUMNS = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

ID_COLUMN = "NUM_CTA_HASH"


# ==========================================================
# FONCTION DE PRÉDICTION
# ==========================================================

def predict_kmeans_2026(
    input_file="scaled_features_2026.csv",
    model_file="kmeans_model_k5.pkl",
    output_file="clients_cluster_2026.csv"
):

    print("\n" + "=" * 60)
    print("PRÉDICTION K-MEANS SUR LES DONNÉES 2026")
    print("=" * 60)

    # ======================================================
    # CHARGEMENT DES DONNÉES 2026
    # ======================================================

    features_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / input_file
    )

    if not features_path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {features_path}"
        )

    df = pd.read_csv(features_path)

    print(
        f"\nFichier 2026 utilisé : {features_path}"
    )

    print(
        f"Nombre de lignes : {len(df)}"
    )

    print(
        f"Nombre de colonnes : {len(df.columns)}"
    )

    # ======================================================
    # VÉRIFICATION DES COLONNES
    # ======================================================

    required_columns = [
        ID_COLUMN
    ] + FEATURE_COLUMNS

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
    # VÉRIFICATION DES IDENTIFIANTS
    # ======================================================

    if df[ID_COLUMN].isnull().any():

        raise ValueError(
            "Certains clients 2026 n'ont pas de "
            "NUM_CTA_HASH."
        )

    if df[ID_COLUMN].duplicated().any():

        duplicates = (
            df[ID_COLUMN]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"{duplicates} identifiants clients "
            "sont dupliqués."
        )

    print(
        f"\nNombre d'identifiants uniques : "
        f"{df[ID_COLUMN].nunique()}"
    )

    # ======================================================
    # CHARGEMENT DU MODÈLE HISTORIQUE
    # ======================================================

    model_path = (
        BASE_DIR
        / "models"
        / model_file
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    model = joblib.load(model_path)

    print(
        f"\nModèle utilisé : {model_path}"
    )

    print(
        f"Nombre de clusters : "
        f"{model.n_clusters}"
    )

    # ======================================================
    # PRÉPARATION DES VARIABLES
    # ======================================================

    features = df[
        FEATURE_COLUMNS
    ].copy()

    print("\nVariables utilisées :")
    print(FEATURE_COLUMNS)

    # ======================================================
    # PRÉDICTION
    # ======================================================

    print("\nApplication du modèle historique...")

    labels = model.predict(features)

    # ======================================================
    # CRÉATION DU RÉSULTAT
    # ======================================================

    result = df[
        [ID_COLUMN]
    ].copy()

    for column in FEATURE_COLUMNS:

        result[column] = df[column].values

    result["CLUSTER"] = labels

    # ======================================================
    # VÉRIFICATION
    # ======================================================

    print("\nNombre de clients par cluster :")

    print(
        result["CLUSTER"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nNombre total de clients : "
        f"{len(result)}"
    )

    print(
        f"Nombre d'identifiants uniques : "
        f"{result[ID_COLUMN].nunique()}"
    )

    # ======================================================
    # SAUVEGARDE
    # ======================================================

    output_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / output_file
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFichier enregistré : "
        f"{output_path}"
    )

    print("\n" + "=" * 60)
    print("PRÉDICTION 2026 TERMINÉE")
    print("=" * 60)

    return result


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    predict_kmeans_2026()