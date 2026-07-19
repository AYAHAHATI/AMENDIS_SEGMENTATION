from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
SAMPLE_DIR = BASE_DIR / "data" / "sample"

SAMPLE_DIR.mkdir(exist_ok=True)

FILES = [
    "HIST_CSO.csv",
    "HIST_CSO_STG22_24.csv",
    "FACT_STG.csv"
]

N_ROWS = 200000

for file in FILES:

    print("=" * 60)
    print(file)

    df = pd.read_csv(
        RAW_DIR / file,
        sep="\t",
        encoding="cp1252",
        low_memory=False,
        nrows=N_ROWS
    )

    output = SAMPLE_DIR / file.replace(".csv", "_SAMPLE.csv")

    df.to_csv(output, index=False)

    print("Créé :", output)
    print(df.shape)

print("\nTous les fichiers SAMPLE sont créés.")