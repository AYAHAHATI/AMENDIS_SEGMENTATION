"""
create_sample_2026.py

Crée un fichier 2026 de TEST (data/sample/HIST_CSO_STG2026_SAMPLE.txt)
à partir de l'échantillon HIST_CSO_SAMPLE.csv : les relevés de 2025
sont décalés d'un an et enregistrés au format du vrai fichier 2026
(tabulation, encodage cp1252).

Ces données ne sont PAS de vraies données 2026 : elles servent
uniquement à tester le pipeline 2026 (DAG, data lake) en mode sample.

Usage :
    python scripts/create_sample_2026.py
    python main.py --sample
    AMENDIS_DATA_MODE=sample python run_2026.py
"""

from pathlib import Path

import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SOURCE = SAMPLE_DIR / "HIST_CSO_SAMPLE.csv"
OUTPUT = SAMPLE_DIR / "HIST_CSO_STG2026_SAMPLE.txt"


def main():
    df = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    dates = pd.to_datetime(df["DAT_PRE_RLV_CSO"], format="%d/%m/%Y", errors="coerce")
    keep = dates.dt.year == 2025
    df = df[keep].copy()
    df["DAT_PRE_RLV_CSO"] = (
        dates[keep] + pd.DateOffset(years=1)
    ).dt.strftime("%d/%m/%Y")
    df.to_csv(OUTPUT, sep="\t", index=False, encoding="cp1252")
    print(f"Créé : {OUTPUT} ({len(df):,} lignes)")


if __name__ == "__main__":
    main()
