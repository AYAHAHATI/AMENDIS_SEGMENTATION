"""
demo_quality.py

Démonstration des contrôles qualité : on prend les vraies données 2026
(silver, réseau courant), on les ABÎME volontairement, puis on lance les
contrôles pour vérifier qu'ils détectent le problème et arrêtent le
pipeline.

Rien n'est modifié sur le disque : les données abîmées restent en
mémoire et les rapports de démonstration ne sont pas enregistrés.

Scénarios :
1. le mois d'avril est absent (fichier tronqué)
2. 5 % des volumes sont négatifs
3. 60 % des contrats ont disparu (mauvais fichier, export partiel)
4. des profils sont en double

Usage :
    python scripts/demo_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.config import (
    APPLICATION_YEAR, SEGMENT_WINDOW, INTERMEDIATE_DIR, ID_COLUMN,
    VOLUME_COLUMN, DATE_COLUMN, window_dates,
)
from pipeline.lake import read_silver
from pipeline.quality import (
    QualityReport, DataQualityError, check_readings, check_profiles,
)


def run(title, check, data, *args):
    print("\n" + "=" * 70)
    print(f"SCÉNARIO : {title}")
    print("=" * 70)
    report = QualityReport("demo")
    check(data, report, *args)
    try:
        report.finish(save=False)
        print("=> Le pipeline CONTINUE")
    except DataQualityError as error:
        print(f"=> PIPELINE ARRÊTÉ : {error}")


def main():
    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    readings = read_silver(start_date=start, end_date=end)
    previous_start, previous_end = window_dates(APPLICATION_YEAR - 1, SEGMENT_WINDOW)
    reference = read_silver(start_date=previous_start, end_date=previous_end)

    run("données normales", check_readings, readings, start, end, reference)

    no_april = readings[readings[DATE_COLUMN].dt.month != 4]
    run("le mois d'avril est absent", check_readings, no_april, start, end, reference)

    negative = readings.copy()
    rows = negative.sample(frac=0.05, random_state=0).index
    negative.loc[rows, VOLUME_COLUMN] = -negative.loc[rows, VOLUME_COLUMN].abs() - 1
    run("5 % des volumes sont négatifs", check_readings, negative, start, end, reference)

    contracts = readings[ID_COLUMN].drop_duplicates()
    kept = contracts.sample(frac=0.4, random_state=0)
    half = readings[readings[ID_COLUMN].isin(kept)]
    run("60 % des contrats ont disparu", check_readings, half, start, end, reference)

    profiles = pd.read_csv(INTERMEDIATE_DIR / "features_2026.csv")
    doubled = pd.concat([profiles, profiles.head(100)], ignore_index=True)
    run("100 profils en double", check_profiles, doubled)


if __name__ == "__main__":
    main()
