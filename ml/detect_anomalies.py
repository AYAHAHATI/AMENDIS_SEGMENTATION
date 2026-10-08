"""
detect_anomalies.py

Détection des anomalies des clients Amendis
avec le modèle Isolation Forest.

Données historiques : 2022-2025

Le modèle entraîné sur les données historiques
est sauvegardé afin de pouvoir être réutilisé
ultérieurement sur les données 2026.
"""

import os

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = (
    "data/final/"
    "clients_segmentes.csv"
)

OUTPUT_FILE = (
    "data/final/"
    "anomaly_scores_historical.csv"
)

MODEL_FILE = (
    "data/final/"
    "isolation_forest_model.joblib"
)

SCALER_FILE = (
    "data/final/"
    "isolation_forest_scaler.joblib"
)


FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]


# ==========================================================
# DÉTECTION DES ANOMALIES
# ==========================================================

def detect_anomalies():

    print("=" * 70)
    print("DÉTECTION DES ANOMALIES - ISOLATION FOREST")
    print("=" * 70)

    # ------------------------------------------------------
    # 1. Chargement des données historiques
    # ------------------------------------------------------

    print("\nChargement des données historiques 2022-2025...")

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Nombre de clients : {len(df)}"
    )

    # ------------------------------------------------------
    # 2. Vérification des features
    # ------------------------------------------------------

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Features manquantes : "
            + str(missing_features)
        )

    print("\nFeatures utilisées :")
    print(FEATURES)

    # ------------------------------------------------------
    # 3. Préparation des données
    # ------------------------------------------------------

    X = df[FEATURES].copy()

    missing_values = X.isnull().sum()

    if missing_values.sum() > 0:

        print("\nValeurs manquantes détectées :")
        print(missing_values)

        raise ValueError(
            "Des valeurs manquantes sont présentes "
            "dans les features."
        )

    print("\nAucune valeur manquante détectée.")

    # ------------------------------------------------------
    # 4. Standardisation
    # ------------------------------------------------------

    print("\nStandardisation...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Standardisation terminée.")

    # ------------------------------------------------------
    # 5. Entraînement Isolation Forest
    # ------------------------------------------------------

    print("\nEntraînement Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    print("Modèle entraîné.")

    # ------------------------------------------------------
    # 6. Détection
    # ------------------------------------------------------

    print("\nDétection des anomalies...")

    predictions = model.predict(X_scaled)

    scores = model.decision_function(X_scaled)

    # ------------------------------------------------------
    # 7. Ajout des résultats
    # ------------------------------------------------------

    df_result = df.copy()

    df_result["ANOMALIE_ISOLATION_FOREST"] = (
        predictions == -1
    )

    df_result["ANOMALY_SCORE"] = scores

    # ------------------------------------------------------
    # 8. Statistiques
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

    print("\n" + "=" * 70)
    print("RÉSULTATS")
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
    # 9. Sauvegarde du dataset avec les anomalies
    # ------------------------------------------------------

    df_result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset des anomalies généré :")
    print(OUTPUT_FILE)

    # ------------------------------------------------------
    # 10. Sauvegarde du modèle
    # ------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModèle Isolation Forest sauvegardé :")
    print(MODEL_FILE)

    # ------------------------------------------------------
    # 11. Sauvegarde du scaler
    # ------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print("\nScaler sauvegardé :")
    print(SCALER_FILE)

    # ------------------------------------------------------
    # 12. Résumé
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("DÉTECTION TERMINÉE")
    print("=" * 70)

    print(
        "Le modèle historique 2022-2025 est prêt "
        "à être réutilisé sur les données 2026."
    )

    return df_result


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    detect_anomalies()