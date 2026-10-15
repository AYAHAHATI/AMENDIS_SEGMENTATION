"""
lake.py

Data Lake du projet Amendis : couches BRONZE, SILVER et GOLD, au
format Parquet.

    fichiers sources (CSV / TXT)
            │  ingest_bronze()      copie fidèle, toutes colonnes en texte
            ▼
    bronze/<fichier>/<fichier>.parquet
            │  build_silver_*()     nettoyage (mêmes règles qu'avant)
            ▼
    silver/releves/reseau=<réseau>/annee=<année>/*.parquet
            │  read_silver()        lecture d'un réseau et d'une fenêtre
            ▼
    profils, segments, anomalies, prédictions
            │  write_gold()
            ▼
    gold/<table>/reseau=<réseau>/part-0.parquet

Pourquoi Parquet ?
- stockage par colonnes : on ne lit que les colonnes utiles ;
- compression : fichiers plusieurs fois plus petits que le CSV ;
- types conservés : une date reste une date, un volume reste un nombre.

Pourquoi des partitions ?
Les relevés silver sont rangés dans un dossier par réseau et par année.
Lire « l'eau de 2024 » n'ouvre que le dossier reseau=eau/annee=2024.

Idempotence : relancer une étape donne le même résultat, sans doublons.
Chaque écriture remplace la partition ou le fichier qu'elle produit.

Usage (ligne de commande) :
    python -m pipeline.lake history   # bronze + silver de l'historique 2022-2025
    python -m pipeline.lake 2026      # bronze + silver des données 2026
    python -m pipeline.lake status    # contenu du lake
"""

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config.config import (
    SOURCE_DIR,
    HIST_FILES,
    FILE_2026,
    APPLICATION_YEAR,
    BRONZE_DIR,
    SILVER_READINGS_DIR,
    GOLD_DIR,
    ID_COLUMN,
    VOLUME_COLUMN,
    DATE_COLUMN,
    NETWORK_COLUMN,
    NETWORK,
    network_slug,
)
from pipeline.transform import clean_consumption

CHUNKSIZE = 500_000
ENCODING = "cp1252"

# Colonnes techniques ajoutées en bronze (traçabilité)
SOURCE_FILE_COLUMN = "_FICHIER_SOURCE"
INGESTION_COLUMN = "_DATE_INGESTION"
METADATA_COLUMNS = [SOURCE_FILE_COLUMN, INGESTION_COLUMN]

# Partitionnement de la couche silver
PARTITION_SCHEMA = pa.schema([("reseau", pa.string()), ("annee", pa.int32())])


# ==========================================================
# OUTILS
# ==========================================================

def detect_separator(path):
    """Détecte le séparateur (tabulation ou virgule) sur la 1re ligne."""
    with open(path, "r", encoding=ENCODING, errors="replace") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def folder_size(path):
    """Taille d'un fichier ou d'un dossier, en octets."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def bronze_name(filename):
    """Nom du jeu bronze d'un fichier source : HIST_CSO.csv -> hist_cso."""
    return Path(filename).stem.lower()


def bronze_path(filename):
    name = bronze_name(filename)
    return BRONZE_DIR / name / f"{name}.parquet"


def source_path(filename):
    """Chemin d'un fichier source (un chemin absolu est accepté)."""
    return SOURCE_DIR / filename


# ==========================================================
# BRONZE : copie fidèle des fichiers sources
# ==========================================================

def ingest_bronze(filename, chunksize=CHUNKSIZE):
    """
    Copie un fichier source en Parquet, SANS le modifier :
    toutes les colonnes, en texte, plus deux colonnes de traçabilité
    (fichier d'origine et date d'ingestion).

    Le fichier est lu par blocs, pour ne pas saturer la mémoire.
    Le résultat remplace l'ancienne version (idempotence).
    """
    path = source_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {path}")

    output = bronze_path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")

    sep = detect_separator(path)
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.time()
    print(f"[bronze] {path.name} -> {output.relative_to(BRONZE_DIR.parent)}")

    writer, schema, rows = None, None, 0
    try:
        for chunk in pd.read_csv(
            path,
            sep=sep,
            dtype=str,              # aucune conversion : copie fidèle
            keep_default_na=False,  # une cellule vide reste vide
            chunksize=chunksize,
            encoding=ENCODING,
            encoding_errors="replace",
        ):
            chunk[SOURCE_FILE_COLUMN] = path.name
            chunk[INGESTION_COLUMN] = ingested_at
            if schema is None:
                schema = pa.schema([(c, pa.string()) for c in chunk.columns])
                writer = pq.ParquetWriter(temporary, schema, compression="snappy")
            writer.write_table(
                pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            )
            rows += len(chunk)
            print(f"  ... {rows:,} lignes", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        raise ValueError(f"Fichier source vide : {path}")
    temporary.replace(output)  # remplacement en une fois

    report = {
        "fichier_source": path.name,
        "lignes": rows,
        "colonnes": len(schema) - len(METADATA_COLUMNS),
        "taille_source_octets": folder_size(path),
        "taille_parquet_octets": folder_size(output),
        "duree_secondes": round(time.time() - started, 1),
        "date_ingestion": ingested_at,
    }
    report["taux_compression"] = round(
        report["taille_source_octets"] / max(report["taille_parquet_octets"], 1), 1
    )
    with open(output.parent / "_ingestion.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"  {rows:,} lignes | CSV {report['taille_source_octets'] / 1e6:,.1f} Mo"
        f" -> Parquet {report['taille_parquet_octets'] / 1e6:,.1f} Mo"
        f" (x{report['taux_compression']})"
    )
    return report


def read_bronze_batches(filename, columns=None, batch_size=CHUNKSIZE):
    """Lit un jeu bronze par blocs (DataFrames pandas)."""
    path = bronze_path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Bronze introuvable : {path} (lancer l'ingestion bronze)"
        )
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


# ==========================================================
# SILVER : relevés nettoyés, partitionnés par réseau et année
# ==========================================================

def write_silver(readings, years):
    """
    Écrit des relevés nettoyés dans silver/releves, partitionnés par
    réseau et par année. Les partitions écrites remplacent les
    anciennes ; les autres années ne sont pas touchées.
    """
    readings = readings.copy()
    readings["reseau"] = readings.pop(NETWORK_COLUMN).map(network_slug)
    readings["annee"] = readings[DATE_COLUMN].dt.year.astype("int32")

    unexpected = sorted(set(readings["annee"]) - set(years))
    if unexpected:
        raise ValueError(f"Années inattendues dans les relevés : {unexpected}")

    SILVER_READINGS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(readings, preserve_index=False)
    ds.write_dataset(
        table,
        SILVER_READINGS_DIR,
        format="parquet",
        partitioning=ds.partitioning(PARTITION_SCHEMA, flavor="hive"),
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )

    summary = readings.groupby(["reseau", "annee"]).agg(
        RELEVES=(ID_COLUMN, "size"), CONTRATS=(ID_COLUMN, "nunique")
    )
    print(summary.to_string())
    return summary


def clean_bronze(filename, id_column, start, end, deduplicate_keys):
    """
    Applique clean_consumption() à un jeu bronze, pour les deux réseaux.

    deduplicate_keys=True reproduit le traitement historique : doublons
    supprimés sur (contrat, volume, date, réseau) après concaténation.
    deduplicate_keys=False reproduit le traitement 2026 : doublons
    supprimés sur la ligne complète (toutes les colonnes source).
    """
    if deduplicate_keys:
        pieces = []
        for chunk in read_bronze_batches(filename):
            chunk = chunk.drop(columns=METADATA_COLUMNS).replace({"": None})
            cleaned = clean_consumption(
                chunk, id_column=id_column, network=None,
                start_date=start, end_date=end,
                verbose=False, keep_network=True,
            )
            if len(cleaned):
                pieces.append(cleaned)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    df = pd.concat(read_bronze_batches(filename), ignore_index=True)
    df = df.drop(columns=METADATA_COLUMNS).replace({"": None})
    return clean_consumption(
        df, id_column=id_column, network=None,
        start_date=start, end_date=end, keep_network=True,
    )


def build_silver_history():
    """Silver de l'historique 2022-2025 (les deux réseaux)."""
    pieces, years = [], set()
    for filename, id_column, start, end in HIST_FILES:
        print(f"[silver] {filename} ({start} -> {end})")
        pieces.append(clean_bronze(filename, id_column, start, end, True))
        years |= set(range(int(start[:4]), int(end[:4]) + 1))

    readings = pd.concat(pieces, ignore_index=True)
    readings = readings.drop_duplicates(
        subset=[ID_COLUMN, VOLUME_COLUMN, DATE_COLUMN, NETWORK_COLUMN]
    ).reset_index(drop=True)
    if readings.empty:
        raise ValueError("Aucun relevé historique après nettoyage.")
    return write_silver(readings, sorted(years))


def build_silver_2026():
    """Silver des données 2026 (les deux réseaux, toute l'année)."""
    print(f"[silver] {FILE_2026}")
    readings = clean_bronze(
        FILE_2026, ID_COLUMN,
        f"{APPLICATION_YEAR}-01-01", f"{APPLICATION_YEAR}-12-31",
        deduplicate_keys=False,
    )
    if readings.empty:
        raise ValueError("Aucun relevé 2026 après nettoyage.")
    return write_silver(readings, [APPLICATION_YEAR])


def silver_years(network=NETWORK):
    """Années disponibles en silver pour un réseau."""
    folder = SILVER_READINGS_DIR / f"reseau={network_slug(network)}"
    if not folder.exists():
        return []
    return sorted(
        int(p.name.split("=")[1]) for p in folder.glob("annee=*")
        if any(p.glob("*.parquet"))
    )


def read_silver(network=NETWORK, start_date=None, end_date=None):
    """
    Lit les relevés silver d'un réseau, éventuellement restreints à une
    fenêtre de dates. Seules les partitions utiles sont ouvertes.
    """
    if not SILVER_READINGS_DIR.exists():
        raise FileNotFoundError(
            f"Silver introuvable : {SILVER_READINGS_DIR} "
            "(lancer python -m pipeline.lake history / 2026)"
        )

    dataset = ds.dataset(SILVER_READINGS_DIR, format="parquet", partitioning="hive")
    condition = ds.field("reseau") == network_slug(network)
    if start_date is not None and end_date is not None:
        first, last = pd.Timestamp(start_date).year, pd.Timestamp(end_date).year
        condition &= ds.field("annee").isin(list(range(first, last + 1)))

    df = dataset.to_table(
        columns=[ID_COLUMN, VOLUME_COLUMN, DATE_COLUMN], filter=condition
    ).to_pandas()

    if start_date is not None:
        df = df[df[DATE_COLUMN] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[DATE_COLUMN] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


# ==========================================================
# GOLD : tables prêtes à l'usage
# ==========================================================

def gold_path(table, network=NETWORK):
    return GOLD_DIR / table / f"reseau={network_slug(network)}" / "part-0.parquet"


def write_gold(df, table, network=NETWORK):
    """Écrit (ou remplace) la partition d'un réseau d'une table gold."""
    path = gold_path(table, network)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    df.to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"[gold] {table} ({network_slug(network)}) : {len(df):,} lignes")
    return path


def read_gold(table, network=None):
    """Lit une table gold (un réseau, ou tous si network=None)."""
    folder = GOLD_DIR / table
    if not folder.exists():
        raise FileNotFoundError(f"Table gold introuvable : {folder}")
    dataset = ds.dataset(folder, format="parquet", partitioning="hive")
    condition = None
    if network is not None:
        condition = ds.field("reseau") == network_slug(network)
    return dataset.to_table(filter=condition).to_pandas()


# ==========================================================
# ORCHESTRATION SIMPLE
# ==========================================================

def build_history_lake(force=False):
    """Bronze + silver de l'historique. Le bronze existant est réutilisé."""
    for filename, *_ in HIST_FILES:
        if force or not bronze_path(filename).exists():
            ingest_bronze(filename)
        else:
            print(f"[bronze] {filename} : déjà ingéré, réutilisé")
    return build_silver_history()


def build_2026_lake():
    ingest_bronze(FILE_2026)
    return build_silver_2026()


def status():
    """Affiche le contenu du lake (tailles et partitions)."""
    for layer in (BRONZE_DIR, SILVER_READINGS_DIR.parent, GOLD_DIR):
        print(f"\n{layer} ({folder_size(layer) / 1e6:,.1f} Mo)" if layer.exists()
              else f"\n{layer} (vide)")
        if layer.exists():
            for file in sorted(layer.rglob("*.parquet")):
                print(f"  {file.relative_to(layer)}  {file.stat().st_size / 1e6:,.2f} Mo")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "history":
        build_history_lake(force="--force" in sys.argv)
    elif command == "2026":
        build_2026_lake()
    elif command == "status":
        status()
    else:
        print(__doc__)
        sys.exit(1)
