"""
history.py

Chargement de l'historique 2022-2025 et construction des profils
de consommation par fenêtre de 4 mois.

Un "profil" = une ligne par contrat et par fenêtre, avec les
5 variables : CONSO_TOTALE, CONSO_MOYENNE, CONSO_MAX, CONSO_MIN,
NB_RELEVES. La même fonction d'agrégation est utilisée pour
l'historique et pour 2026.
"""

import pandas as pd

from config.config import (
    SOURCE_DIR,
    HIST_FILES,
    ID_COLUMN,
    VOLUME_COLUMN,
    DATE_COLUMN,
    NETWORK_COLUMN,
    FEATURES,
    MIN_RELEVES,
)
from pipeline.transform import clean_consumption


def detect_separator(path):
    """Détecte le séparateur (tabulation ou virgule) sur la 1re ligne."""
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def load_history(chunksize=500_000):
    """
    Charge les relevés historiques 2022-2025 (un seul réseau),
    nettoyés avec les mêmes règles que les données 2026.
    """
    pieces = []

    for filename, id_column, start, end in HIST_FILES:
        path = SOURCE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        sep = detect_separator(path)
        print(f"\nLecture : {filename} (séparateur {sep!r})")

        header = pd.read_csv(
            path, sep=sep, nrows=0, encoding="cp1252",
            encoding_errors="replace",
        ).columns
        usecols = [id_column, VOLUME_COLUMN, DATE_COLUMN]
        if NETWORK_COLUMN in header:
            usecols.append(NETWORK_COLUMN)

        lines_read = 0
        for chunk in pd.read_csv(
            path,
            sep=sep,
            usecols=usecols,
            chunksize=chunksize,
            encoding="cp1252",
            encoding_errors="replace",
            low_memory=False,
        ):
            lines_read += len(chunk)
            print(f"  ... {lines_read:,} lignes lues", flush=True)
            chunk = clean_consumption(
                chunk,
                id_column=id_column,
                start_date=start,
                end_date=end,
                verbose=False,
            )
            if len(chunk):
                pieces.append(chunk)

    if not pieces:
        raise ValueError("Aucun relevé historique après nettoyage.")

    df = pd.concat(pieces, ignore_index=True).drop_duplicates()

    print(
        f"\nRelevés historiques : {len(df):,} | "
        f"contrats : {df[ID_COLUMN].nunique():,} | "
        f"période : {df[DATE_COLUMN].min().date()} -> "
        f"{df[DATE_COLUMN].max().date()}"
    )
    return df


def aggregate_window(df, start_date=None, end_date=None):
    """
    Agrège les relevés d'une fenêtre en un profil par contrat.
    Sans dates : agrège tout le DataFrame (déjà filtré).
    """
    part = df
    if start_date is not None:
        part = part[part[DATE_COLUMN] >= pd.Timestamp(start_date)]
    if end_date is not None:
        part = part[part[DATE_COLUMN] <= pd.Timestamp(end_date)]

    # Un même relevé (contrat + date) apparaît parfois sur plusieurs
    # lignes (plusieurs registres/compteurs, souvent une ligne à 0).
    # On les additionne : un relevé = la consommation de la période.
    readings = (
        part.groupby([ID_COLUMN, DATE_COLUMN], as_index=False)[VOLUME_COLUMN]
        .sum()
    )

    profile = (
        readings.groupby(ID_COLUMN)[VOLUME_COLUMN]
        .agg(
            CONSO_TOTALE="sum",
            CONSO_MOYENNE="mean",
            CONSO_MAX="max",
            CONSO_MIN="min",
            NB_RELEVES="count",
        )
        .reset_index()
    )
    profile = profile[[ID_COLUMN] + FEATURES]
    profile["PROFIL_COMPLET"] = profile["NB_RELEVES"] >= MIN_RELEVES
    return profile
