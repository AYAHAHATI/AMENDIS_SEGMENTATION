"""
combine_reseaux.py

Réunit les résultats 2026 de l'électricité et de l'eau.

1. data/final/tous_reseaux/ : fichiers combinés (nouveau format).
2. data/final/ : fichiers au FORMAT DE L'ANCIENNE VERSION
   (mêmes noms de fichiers, mêmes colonnes, même ordre), pour que le
   tableau de bord Power BI existant se mette à jour avec « Actualiser »
   sans modifier les visuels. Les nouvelles colonnes (RESEAU, ...) sont
   ajoutées À LA FIN, pour ne pas décaler les colonnes existantes.

Chaque réseau a son propre modèle (unités différentes : kWh et m3).
Dans Power BI, filtrer par RESEAU : ne jamais additionner kWh et m3.

Usage (après avoir traité les deux réseaux) :
    python combine_reseaux.py
"""

import numpy as np
import pandas as pd

from config.config import DATA_DIR, COMBINED_DIR, ID_COLUMN

NETWORK_DIRS = ["electricite", "eau"]
FILES = [
    "clients_segmentes_2026.csv",
    "anomaly_scores_2026.csv",
    "predictions_segments_2026.csv",
    "metriques_prediction_2026.csv",
]
LEGACY_DIR = DATA_DIR / "final"

# Colonnes de l'ancienne version, dans l'ordre d'origine
PROFILE_COLUMNS = [
    ID_COLUMN, "CONSO_TOTALE", "CONSO_MOYENNE", "CONSO_MAX",
    "CONSO_MIN", "NB_RELEVES", "CLUSTER", "LIBELLE_CLUSTER",
]
LEGACY_ANOMALY_COLUMNS = PROFILE_COLUMNS + [
    "ANOMALIE", "SCORE_ANOMALIE", "EST_ANOMALIE",
]
LEGACY_PREDICTION_COLUMNS = [
    ID_COLUMN, "SEGMENT_PREDIT_2026", "CLUSTER_REEL_2026",
    "SEGMENT_REEL_2026", "CORRECT",
]
LEGACY_METRIC_COLUMNS = [
    "MODELE", "CLIENTS_EVALUES", "ACCURACY", "PRECISION_WEIGHTED",
    "RECALL_WEIGHTED", "F1_WEIGHTED", "PRECISION_MACRO",
    "RECALL_MACRO", "F1_MACRO",
]


def ordered(df, first_columns):
    """Anciennes colonnes d'abord, nouvelles colonnes à la fin."""
    rest = [c for c in df.columns if c not in first_columns]
    return df[first_columns + rest]


def combine():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combined = {}

    for filename in FILES:
        parts = []
        for network_dir in NETWORK_DIRS:
            path = DATA_DIR / "final" / network_dir / filename
            if path.exists():
                parts.append(pd.read_csv(path))
            else:
                print(f"ATTENTION : {path} absent (réseau non traité ?)")
        if not parts:
            continue

        df = pd.concat(parts, ignore_index=True)
        combined[filename] = df
        output = COMBINED_DIR / filename
        df.to_csv(output, index=False)
        print(f"{output} : {len(df):,} lignes {df['RESEAU'].value_counts().to_dict()}")

    return combined


def export_legacy(combined):
    """Fichiers au format de l'ancienne version, dans data/final/."""
    print("\nExport au format de l'ancien tableau de bord Power BI (data/final/) :")

    clients = combined["clients_segmentes_2026.csv"]
    clients = ordered(clients, PROFILE_COLUMNS)
    clients.to_csv(LEGACY_DIR / "clients_segmentes_2026.csv", index=False)

    anomalies = combined["anomaly_scores_2026.csv"].copy()
    # Ancienne colonne ANOMALIE : -1 = anomalie, 1 = normal (vide = non évalué)
    anomalies["ANOMALIE"] = np.where(
        anomalies["EST_ANOMALIE"].isna(), np.nan,
        np.where(anomalies["EST_ANOMALIE"] == 1, -1, 1),
    )
    # Entiers comme dans l'ancienne version (« 1 » et non « 1.0 »)
    for column in ("ANOMALIE", "EST_ANOMALIE", "CLUSTER", "NB_RELEVES"):
        anomalies[column] = anomalies[column].astype("Int64")
    anomalies = ordered(anomalies, LEGACY_ANOMALY_COLUMNS)
    anomalies.to_csv(LEGACY_DIR / "anomaly_scores_2026.csv", index=False)

    predictions = combined["predictions_segments_2026.csv"].rename(
        columns={"SEGMENT_REFERENCE_2026": "SEGMENT_REEL_2026"}
    )
    predictions = predictions.merge(
        clients[[ID_COLUMN, "RESEAU", "CLUSTER"]].rename(
            columns={"CLUSTER": "CLUSTER_REEL_2026"}
        ),
        on=[ID_COLUMN, "RESEAU"], how="left",
    )
    predictions["CLUSTER_REEL_2026"] = predictions["CLUSTER_REEL_2026"].astype("Int64")
    predictions = ordered(predictions, LEGACY_PREDICTION_COLUMNS)
    predictions.to_csv(
        LEGACY_DIR / "prediction_segments_2026.csv",
        index=False, encoding="utf-8-sig",
    )

    # Ancien fichier : une ligne par modèle retenu (sans les baselines,
    # qui restent dans tous_reseaux/metriques_prediction_2026.csv)
    metrics = combined["metriques_prediction_2026.csv"]
    metrics = metrics[~metrics["MODELE"].str.startswith("Baseline")]
    metrics = metrics.rename(columns={"CONTRATS_EVALUES": "CLIENTS_EVALUES"})
    metrics = ordered(metrics, LEGACY_METRIC_COLUMNS)
    metrics.to_csv(
        LEGACY_DIR / "prediction_2026_metrics.csv",
        index=False, encoding="utf-8-sig",
    )

    # Ancienne matrice de confusion (lignes « REEL: », colonnes « PREDIT: »)
    labels = sorted(
        set(predictions["SEGMENT_REEL_2026"]) | set(predictions["SEGMENT_PREDIT_2026"])
    )
    blocks = []
    for network, part in predictions.groupby("RESEAU"):
        matrix = pd.crosstab(
            part["SEGMENT_REEL_2026"], part["SEGMENT_PREDIT_2026"]
        ).reindex(index=labels, columns=labels, fill_value=0)
        matrix.index = [f"REEL: {label}" for label in labels]
        matrix.columns = [f"PREDIT: {label}" for label in labels]
        matrix["RESEAU"] = network
        blocks.append(matrix)
    pd.concat(blocks).to_csv(
        LEGACY_DIR / "prediction_segments_2026_confusion.csv",
        encoding="utf-8-sig",
    )

    for name in [
        "clients_segmentes_2026.csv", "anomaly_scores_2026.csv",
        "prediction_segments_2026.csv", "prediction_2026_metrics.csv",
        "prediction_segments_2026_confusion.csv",
    ]:
        print(f"  {LEGACY_DIR / name}")


if __name__ == "__main__":
    data = combine()
    if all(name in data for name in FILES):
        export_legacy(data)
    else:
        print("\nExport Power BI non réalisé : traiter d'abord les deux réseaux.")
