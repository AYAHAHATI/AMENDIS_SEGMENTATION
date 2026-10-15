"""
detect_anomalies.py

Entraîne la détection des comportements atypiques sur les profils
janvier-avril 2022-2025, avec un Isolation Forest PAR SEGMENT
(200 arbres, contamination de 5 %, random_state = 42).

Pourquoi par segment ?
Un modèle global, entraîné sur tous les contrats, signale surtout les
plus gros consommateurs : ils sont "rares" uniquement parce qu'ils
consomment beaucoup (100 % des segments Forte et Très forte étaient
signalés). Avec un modèle par segment, un contrat est atypique s'il
s'écarte des contrats de SON niveau de consommation.

Un segment trop petit (< MIN_SEGMENT_SIZE_FOR_IF profils) n'a pas
de modèle : ses contrats sont marqués "non évalué".

Les variables sont celles de la segmentation (MODEL_FEATURES),
standardisées avec le scaler historique.

Sortie : models/<reseau>/isolation_forest_models.joblib
         (dictionnaire {numéro de cluster: modèle})
"""

import joblib
import numpy as np
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
    MIN_SEGMENT_SIZE_FOR_IF,
    RANDOM_STATE,
    ensure_dirs,
)

INPUT_FILE = INTERMEDIATE_DIR / f"clients_cluster_k{N_CLUSTERS}.csv"


def train_anomaly_models():
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} introuvable : lancer ml/train_kmeans.py")

    df = pd.read_csv(INPUT_FILE)
    df = df[df["CLUSTER"] >= 0].reset_index(drop=True)  # profils complets
    scaler = joblib.load(SCALER_FILE)
    X = pd.DataFrame(scaler.transform(df[MODEL_FEATURES]), columns=MODEL_FEATURES)

    print(
        f"Isolation Forest par segment : {ISOLATION_FOREST_TREES} arbres, "
        f"contamination {ISOLATION_FOREST_CONTAMINATION:.0%}"
    )
    models = {}
    for cluster, index in df.groupby("CLUSTER").groups.items():
        label = df.loc[index[0], "LIBELLE_CLUSTER"]
        if len(index) >= MIN_SEGMENT_SIZE_FOR_IF:
            models[int(cluster)] = IsolationForest(
                n_estimators=ISOLATION_FOREST_TREES,
                contamination=ISOLATION_FOREST_CONTAMINATION,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ).fit(X.loc[index])
            print(f"  {label} : modèle entraîné ({len(index):,} profils)")
        else:
            print(
                f"  {label} : {len(index):,} profils "
                f"(< {MIN_SEGMENT_SIZE_FOR_IF}) -> non évalué"
            )

    joblib.dump(models, ANOMALY_MODEL_FILE)

    scored = score_profiles(df, X, models)
    print("\nAnomalies sur l'historique par segment :")
    print(summarize(scored).to_string())
    scored.to_csv(FINAL_DIR / "anomaly_scores_historical.csv", index=False)
    from pipeline.lake import write_gold
    write_gold(scored, "anomalies_historique")
    print(f"Modèles enregistrés : {ANOMALY_MODEL_FILE}")
    return models


def score_profiles(df, X_scaled, models):
    """
    Chaque contrat est évalué par le modèle de son segment.
    EST_ANOMALIE : 1 = atypique, 0 = normal, vide = non évalué.
    SCORE_ANOMALIE : plus il est bas, plus le profil est atypique.
    """
    if not isinstance(models, dict):
        raise TypeError(
            "Ancien modèle global détecté : relancer "
            "python -m ml.detect_anomalies pour entraîner les modèles par segment."
        )
    result = df.copy()
    result["EST_ANOMALIE"] = np.nan
    result["SCORE_ANOMALIE"] = np.nan

    for cluster, index in df.groupby("CLUSTER").groups.items():
        model = models.get(int(cluster))
        if model is None:
            continue
        Xc = X_scaled.loc[index]
        result.loc[index, "EST_ANOMALIE"] = (model.predict(Xc) == -1).astype(int)
        result.loc[index, "SCORE_ANOMALIE"] = model.decision_function(Xc)
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
