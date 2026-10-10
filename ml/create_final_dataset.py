"""
create_final_dataset.py

Dataset final 2026 pour Power BI : profil de consommation
(valeurs réelles, non standardisées) + segment K-Means.
"""

import pandas as pd

from config.config import INTERMEDIATE_DIR, FINAL_DIR, ID_COLUMN, NETWORK


def create_final_dataset(
    source_file="features_2026.csv",
    clusters_file="clients_cluster_2026.csv",
    output_file="clients_segmentes_2026.csv",
):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(INTERMEDIATE_DIR / source_file)
    clusters = pd.read_csv(INTERMEDIATE_DIR / clusters_file)

    final = profiles.merge(
        clusters[[ID_COLUMN, "CLUSTER", "LIBELLE_CLUSTER"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(final) != len(profiles) or final["CLUSTER"].isna().any():
        raise ValueError("Certains contrats 2026 n'ont pas de segment.")

    # Réseau, pour combiner électricité et eau dans Power BI
    final.insert(1, "RESEAU", NETWORK)

    output_path = FINAL_DIR / output_file
    final.to_csv(output_path, index=False)

    print(final["LIBELLE_CLUSTER"].value_counts().to_string())
    print(f"Fichier enregistré : {output_path} ({len(final):,} contrats)")
    return final


if __name__ == "__main__":
    create_final_dataset()
