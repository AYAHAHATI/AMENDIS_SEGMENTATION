"""
combine_reseaux.py

Réunit les résultats 2026 de l'électricité et de l'eau dans
data/final/tous_reseaux/, pour Power BI.

Chaque réseau a son propre modèle (unités différentes : kWh et m3).
Les fichiers combinés gardent la colonne RESEAU : dans Power BI,
filtrer ou découper par RESEAU, et ne jamais additionner des
consommations de réseaux différents.

Usage (après avoir traité les deux réseaux) :
    python combine_reseaux.py
"""

import pandas as pd

from config.config import DATA_DIR, COMBINED_DIR

NETWORK_DIRS = ["electricite", "eau"]
FILES = [
    "clients_segmentes_2026.csv",
    "anomaly_scores_2026.csv",
    "predictions_segments_2026.csv",
    "metriques_prediction_2026.csv",
]


def combine():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

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

        combined = pd.concat(parts, ignore_index=True)
        output = COMBINED_DIR / filename
        combined.to_csv(output, index=False)

        summary = (
            combined["RESEAU"].value_counts().to_dict()
            if "RESEAU" in combined.columns else {}
        )
        print(f"{output} : {len(combined):,} lignes {summary}")


if __name__ == "__main__":
    combine()
