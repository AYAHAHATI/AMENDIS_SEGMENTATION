"""
build_training_windows.py

Construit les profils d'entraînement : un profil par contrat et
par fenêtre janvier-avril, pour chaque année 2022 à 2025.

Résultat : data/intermediate/training_windows.csv
(colonnes : NUM_CTA_HASH, ANNEE, 5 variables)

Pourquoi ? Les données 2026 couvrent janvier-avril (mai est
incomplet). Le modèle doit donc apprendre sur des profils de
même durée, sinon les totaux et le nombre de relevés ne sont
pas comparables.
"""

import pandas as pd

from config.config import (
    INTERMEDIATE_DIR,
    TRAINING_YEARS,
    SEGMENT_WINDOW,
    window_dates,
    ensure_dirs,
)
from pipeline.history import load_history, aggregate_window

OUTPUT_FILE = INTERMEDIATE_DIR / "training_windows.csv"


def build_training_windows(history=None):
    ensure_dirs()
    if history is None:
        history = load_history()

    profiles = []
    for year in TRAINING_YEARS:
        start, end = window_dates(year, SEGMENT_WINDOW)
        profile = aggregate_window(history, start, end)
        profile.insert(1, "ANNEE", year)
        print(f"Fenêtre {start} -> {end} : {len(profile):,} contrats")
        profiles.append(profile)

    result = pd.concat(profiles, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nProfils d'entraînement : {len(result):,}")
    print(result.drop(columns=["ANNEE"]).describe().round(2))
    print(f"Fichier enregistré : {OUTPUT_FILE}")
    return result


if __name__ == "__main__":
    build_training_windows()
