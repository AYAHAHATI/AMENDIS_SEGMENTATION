"""
detect_anomalies.py

Entraîne l'Isolation Forest sur les profils janvier-avril 2022-2025,
comme décrit dans le rapport : un modèle global, 200 arbres,
contamination de 5 %, random_state = 42.

Les variables sont celles de la segmentation (MODEL_FEATURES),
standardisées avec le scaler historique.

Limite connue (à discuter dans le rapport) : un modèle global signale
en priorité les plus gros consommateurs, car ils sont rares.
Le taux d'anomalies par segment est affiché pour le montrer.

Sortie : models/isolation_forest_models.joblib
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from config.config import (
    INTERMEDIATE_DIR,
    FINAL_DIR,
    MODEL_FEATURES,
    N_CLUSTERS,
    SCALER_FILE,
    ANOMALY_MODEL_FILE,
    ISOLATION_FOREST_TREES,
    ISOLATION_FOREST_CONTAMINATION,
    RANDOM_STATE,
    ensure_dirs,
)

INPUT_FILE = INTERMEDIATE_DIR / f"clients_cluster_k{N_CLUSTERS}.csv"


def train_anomaly_models():
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} introuvable : lancer ml/train_kmeans.py")

    df = pd.read_csv(INPUT_FILE)
    scaler = joblib.load(SCALER_FILE)
    X = pd.DataFrame(scaler.transform(df[MODEL_FEATURES]), columns=MODEL_FEATURES)

    model = IsolationForest(
        n_estimators=ISOLATION_FOREST_TREES,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X)
    joblib.dump(model, ANOMALY_MODEL_FILE)

    scored = score_profiles(df, X, model)
    print(
        f"Isolation Forest : {ISOLATION_FOREST_TREES} arbres, "
        f"contamination {ISOLATION_FOREST_CONTAMINATION:.0%}, "
        f"{len(df):,} profils"
    )
    print("\nAnomalies sur l'historique par segment :")
    print(summarize(scored).to_string())
    scored.to_csv(FINAL_DIR / "anomaly_scores_historical.csv", index=False)
    print(f"Modèle enregistré : {ANOMALY_MODEL_FILE}")
    return model


def score_profiles(df, X_scaled, model):
    """
    EST_ANOMALIE : 1 = atypique, 0 = normal.
    SCORE_ANOMALIE : plus il est bas, plus le profil est atypique.
    """
    result = df.copy()
    result["EST_ANOMALIE"] = (model.predict(X_scaled) == -1).astype(int)
    result["SCORE_ANOMALIE"] = model.decision_function(X_scaled)
    return result


def summarize(scored):
    summary = scored.groupby("LIBELLE_CLUSTER").agg(
        CONTRATS=("EST_ANOMALIE", "size"),
        EVALUES=("EST_ANOMALIE", "count"),
        ANOMALIES=("EST_ANOMALIE", "sum"),
    )
    summary["TAUX_%"] = (summary["ANOMALIES"] / summary["EVALUES"] * 100).round(2)
    return summary


if __name__ == "__main__":
    train_anomaly_models()
