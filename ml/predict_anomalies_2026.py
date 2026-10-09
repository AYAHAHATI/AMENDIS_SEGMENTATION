"""
predict_anomalies_2026.py

Applique les Isolation Forest historiques (sans réentraînement)
aux profils janvier-avril 2026, segment par segment.

Une "anomalie" est un profil signalé par le modèle selon le seuil
de contamination appris sur l'historique ; ce n'est ni une fraude
ni une erreur confirmée.
"""

import joblib
import pandas as pd

from config.config import FINAL_DIR, MODEL_FEATURES, SCALER_FILE, ANOMALY_MODEL_FILE
from ml.detect_anomalies import score_profiles, summarize


def predict_anomalies_2026(
    input_file="clients_segmentes_2026.csv",
    output_file="anomaly_scores_2026.csv",
):
    input_path = FINAL_DIR / input_file
    for path in (input_path, SCALER_FILE, ANOMALY_MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(input_path)
    complete = df[df["CLUSTER"] >= 0].reset_index(drop=True)
    incomplete = df[df["CLUSTER"] < 0]

    scaler = joblib.load(SCALER_FILE)
    models = joblib.load(ANOMALY_MODEL_FILE)
    X = pd.DataFrame(
        scaler.transform(complete[MODEL_FEATURES]), columns=MODEL_FEATURES
    )

    scored = score_profiles(complete, X, models)
    # Profils incomplets : non évalués
    result = pd.concat([scored, incomplete], ignore_index=True)

    n = int(result["EST_ANOMALIE"].count())
    n_anomalies = int(result["EST_ANOMALIE"].sum())
    print(
        f"Contrats : {len(result):,} | évalués : {n:,} | anomalies : "
        f"{n_anomalies:,} ({n_anomalies / max(n, 1):.2%} des évalués)"
    )
    print(summarize(result).to_string())

    output_path = FINAL_DIR / output_file
    result.to_csv(output_path, index=False)
    print(f"Fichier enregistré : {output_path}")
    return result


if __name__ == "__main__":
    predict_anomalies_2026()
