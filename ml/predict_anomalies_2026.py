"""
predict_anomalies_2026.py

Application du modèle Isolation Forest historique
sur les données 2026.

Le modèle a été entraîné sur les données historiques 2022-2025.
Les données 2026 sont utilisées uniquement comme données de test.

Aucun réentraînement du modèle n'est effectué sur 2026.
"""

import os

import joblib
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = (
    "/opt/airflow/data/final/"
    "clients_segmentes_2026.csv"
)

MODEL_FILE = (
    "/opt/airflow/data/final/"
    "isolation_forest_model.joblib"
)

SCALER_FILE = (
    "/opt/airflow/data/final/"
    "isolation_forest_scaler.joblib"
)

OUTPUT_FILE = (
    "/opt/airflow/data/final/"
    "anomaly_scores_2026.csv"
)


FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]


# ==========================================================
# PRÉDICTION DES ANOMALIES 2026
# ==========================================================

def predict_anomalies_2026():

    print("=" * 70)
    print("DÉTECTION DES ANOMALIES 2026")
    print("MODÈLE : ISOLATION FOREST")
    print("=" * 70)

    # ------------------------------------------------------
    # 1. Vérification des fichiers
    # ------------------------------------------------------

    print("\nVérification des fichiers...")

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Dataset 2026 introuvable : {INPUT_FILE}"
        )

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Modèle Isolation Forest introuvable : {MODEL_FILE}"
        )

    if not os.path.exists(SCALER_FILE):

        raise FileNotFoundError(
            f"Scaler historique introuvable : {SCALER_FILE}"
        )

    print("Tous les fichiers nécessaires sont disponibles.")

    # ------------------------------------------------------
    # 2. Chargement des données 2026
    # ------------------------------------------------------

    print("\nChargement des données 2026...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Nombre de clients 2026 : {len(df)}"
    )

    # ------------------------------------------------------
    # 3. Vérification des features
    # ------------------------------------------------------

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Features manquantes dans les données 2026 : "
            + str(missing_features)
        )

    print("\nFeatures utilisées :")
    print(FEATURES)

    # ------------------------------------------------------
    # 4. Préparation des données
    # ------------------------------------------------------

    X = df[FEATURES].copy()

    missing_values = X.isnull().sum()

    if missing_values.sum() > 0:

        print("\nValeurs manquantes détectées :")
        print(missing_values)

        raise ValueError(
            "Des valeurs manquantes sont présentes "
            "dans les features 2026."
        )

    print("\nAucune valeur manquante détectée.")

    # ------------------------------------------------------
    # 5. Chargement du scaler historique
    # ------------------------------------------------------

    print("\nChargement du scaler historique 2022-2025...")

    scaler = joblib.load(SCALER_FILE)

    print("Scaler historique chargé.")

    # ------------------------------------------------------
    # 6. Standardisation 2026
    # ------------------------------------------------------

    print("\nStandardisation des données 2026...")

    X_scaled = scaler.transform(X)

    print(
        "Standardisation terminée avec le scaler "
        "entraîné sur 2022-2025."
    )

    # ------------------------------------------------------
    # 7. Chargement du modèle historique
    # ------------------------------------------------------

    print(
        "\nChargement du modèle Isolation Forest "
        "historique 2022-2025..."
    )

    model = joblib.load(MODEL_FILE)

    print("Modèle historique chargé.")

    # ------------------------------------------------------
    # 8. Prédiction
    # ------------------------------------------------------

    print("\nDétection des anomalies 2026...")

    predictions = model.predict(X_scaled)

    scores = model.decision_function(X_scaled)

    # ------------------------------------------------------
    # 9. Ajout des résultats
    # ------------------------------------------------------

    df_result = df.copy()

    df_result["ANOMALIE_ISOLATION_FOREST"] = (
        predictions == -1
    )

    df_result["ANOMALY_SCORE"] = scores

    # ------------------------------------------------------
    # 10. Statistiques
    # ------------------------------------------------------

    nb_anomalies = int(
        df_result[
            "ANOMALIE_ISOLATION_FOREST"
        ].sum()
    )

    nb_normaux = (
        len(df_result)
        - nb_anomalies
    )

    pourcentage = (
        nb_anomalies
        / len(df_result)
        * 100
    )

    # ------------------------------------------------------
    # 11. Affichage des résultats
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉSULTATS - DONNÉES 2026")
    print("=" * 70)

    print(
        f"Nombre total de clients : "
        f"{len(df_result)}"
    )

    print(
        f"Nombre d'anomalies : "
        f"{nb_anomalies}"
    )

    print(
        f"Pourcentage d'anomalies : "
        f"{pourcentage:.2f}%"
    )

    print(
        f"Nombre de clients normaux : "
        f"{nb_normaux}"
    )

    # ------------------------------------------------------
    # 12. Sauvegarde
    # ------------------------------------------------------

    df_result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset 2026 avec détection des anomalies généré :")
    print(OUTPUT_FILE)

    # ------------------------------------------------------
    # 13. Résumé
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRÉDICTION 2026 TERMINÉE")
    print("=" * 70)

    print(
        "Le modèle Isolation Forest entraîné sur "
        "2022-2025 a été appliqué aux données 2026."
    )

    print(
        "Aucun réentraînement n'a été effectué sur 2026."
    )

    return df_result


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    predict_anomalies_2026()