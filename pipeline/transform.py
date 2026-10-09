"""
transform.py

Nettoyage et transformation des relevés de consommation Amendis.

Règles appliquées (identiques pour l'historique et pour 2026) :
- suppression des lignes strictement dupliquées ;
- conversion de VOL_CONSO et de la date de relevé ;
- suppression des relevés sans volume, sans date ou sans identifiant
  (on n'invente pas de consommation : pas d'imputation par la médiane) ;
- conservation d'un seul réseau (électricité OU eau) ;
- filtrage optionnel sur une fenêtre de dates.
"""

import pandas as pd

from config.config import (
    INTERMEDIATE_DIR,
    ID_COLUMN,
    VOLUME_COLUMN,
    DATE_COLUMN,
    NETWORK_COLUMN,
    NETWORK,
)


def parse_dates(series):
    """Convertit les dates jj/mm/aaaa (format Amendis)."""
    dates = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    if dates.isna().mean() > 0.5:
        # Format différent : conversion plus permissive
        dates = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return dates


def clean_consumption(
    df,
    id_column=ID_COLUMN,
    network=NETWORK,
    start_date=None,
    end_date=None,
    verbose=True,
):
    """
    Nettoie un DataFrame de relevés et renvoie les colonnes utiles :
    NUM_CTA_HASH, VOL_CONSO, DAT_PRE_RLV_CSO.
    """
    stats = {"lignes_initiales": len(df)}

    df = df.drop_duplicates()
    stats["apres_doublons"] = len(df)

    if id_column != ID_COLUMN:
        df = df.rename(columns={id_column: ID_COLUMN})

    df = df.copy()
    df[VOLUME_COLUMN] = pd.to_numeric(
        df[VOLUME_COLUMN].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df[DATE_COLUMN] = parse_dates(df[DATE_COLUMN])

    df = df.dropna(subset=[ID_COLUMN, VOLUME_COLUMN, DATE_COLUMN])
    stats["apres_valeurs_manquantes"] = len(df)

    if network and NETWORK_COLUMN in df.columns:
        df = df[
            df[NETWORK_COLUMN].astype(str).str.strip().str.upper()
            == network.upper()
        ]
        stats[f"reseau_{network}"] = len(df)
    elif network:
        print(
            f"ATTENTION : colonne {NETWORK_COLUMN} absente, "
            "pas de filtrage par réseau."
        )

    if start_date is not None:
        df = df[df[DATE_COLUMN] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[DATE_COLUMN] <= pd.Timestamp(end_date)]
    if start_date is not None or end_date is not None:
        stats["dans_la_fenetre"] = len(df)

    if verbose:
        for step, value in stats.items():
            print(f"  {step:<28}: {value:,}")

    return df[[ID_COLUMN, VOLUME_COLUMN, DATE_COLUMN]].reset_index(drop=True)


class DataTransformer:
    """Transformation d'un fichier extrait (utilisée par le DAG 2026)."""

    def run(self, input_file, output_file, start_date=None, end_date=None):
        input_path = INTERMEDIATE_DIR / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

        print(f"Lecture : {input_path}")
        df = pd.read_csv(input_path, low_memory=False)

        df = clean_consumption(
            df, start_date=start_date, end_date=end_date
        )

        if df.empty:
            raise ValueError(
                "Aucun relevé après nettoyage : vérifier le réseau "
                "et la fenêtre de dates."
            )

        print(
            f"Période conservée : {df[DATE_COLUMN].min().date()} -> "
            f"{df[DATE_COLUMN].max().date()}"
        )
        print(df[VOLUME_COLUMN].describe())

        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = INTERMEDIATE_DIR / output_file
        df.to_csv(output_path, index=False)
        print(f"Fichier enregistré : {output_path}")
        return df
