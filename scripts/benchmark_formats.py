"""
benchmark_formats.py

Compare le fichier source (CSV/TXT) et sa copie bronze en Parquet :
- taille sur le disque ;
- temps de lecture des 4 colonnes utiles au pipeline
  (identifiant, volume, date, réseau).

Pré-requis : le bronze existe (python -m pipeline.lake history / 2026).

Usage :
    python scripts/benchmark_formats.py            # données réelles
    AMENDIS_DATA_MODE=sample python scripts/benchmark_formats.py

Résultat : data/lake/benchmark_formats.csv (repris dans le rapport).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq

from config.config import HIST_FILES, FILE_2026, LAKE_DIR, VOLUME_COLUMN, DATE_COLUMN, NETWORK_COLUMN
from pipeline.lake import bronze_path, source_path, detect_separator, folder_size, ENCODING


def timed(function, repeat=2):
    """Meilleur temps sur plusieurs essais (en secondes)."""
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def main():
    rows = []
    sources = [(name, id_column) for name, id_column, *_ in HIST_FILES]
    sources.append((FILE_2026, "NUM_CTA_HASH"))

    for filename, id_column in sources:
        csv_path, parquet_path = source_path(filename), bronze_path(filename)
        if not (csv_path.exists() and parquet_path.exists()):
            print(f"Ignoré (source ou bronze absent) : {filename}")
            continue
        columns = [id_column, VOLUME_COLUMN, DATE_COLUMN, NETWORK_COLUMN]
        sep = detect_separator(csv_path)

        csv_time = timed(lambda: pd.read_csv(
            csv_path, sep=sep, usecols=columns, encoding=ENCODING,
            encoding_errors="replace", low_memory=False,
        ))
        parquet_time = timed(lambda: pq.read_table(parquet_path, columns=columns))

        csv_size, parquet_size = folder_size(csv_path), folder_size(parquet_path)
        rows.append({
            "FICHIER": filename,
            "LIGNES": pq.ParquetFile(parquet_path).metadata.num_rows,
            "TAILLE_CSV_MO": round(csv_size / 1e6, 1),
            "TAILLE_PARQUET_MO": round(parquet_size / 1e6, 1),
            "GAIN_TAILLE": round(csv_size / parquet_size, 1),
            "LECTURE_CSV_S": round(csv_time, 2),
            "LECTURE_PARQUET_S": round(parquet_time, 2),
            "GAIN_LECTURE": round(csv_time / parquet_time, 1),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    LAKE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(LAKE_DIR / "benchmark_formats.csv", index=False)
    print(f"\nEnregistré : {LAKE_DIR / 'benchmark_formats.csv'}")


if __name__ == "__main__":
    main()
