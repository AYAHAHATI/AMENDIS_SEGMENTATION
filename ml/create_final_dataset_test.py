"""
create_final_dataset_test.py

Création du dataset final des clients segmentés
à partir des résultats K-Means K=5.
"""

import pandas as pd
from pathlib import Path


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# FONCTION PRINCIPALE
# ==========================================================

def create_final_dataset_test():

    print("\n" + "=" * 60)
    print("CRÉATION DU DATASET FINAL DE SEGMENTATION")
    print("=" * 60)

    # ======================================================
    # CHARGEMENT DU DATASET SOURCE
    # ======================================================

    dataset_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / "dataset_final.csv"
    )

    dataset = pd.read_csv(dataset_path)

    print(f"\nDataset source : {dataset_path}")
    print(f"Nombre de lignes : {len(dataset)}")

    # ======================================================
    # VÉRIFICATION DE L'IDENTIFIANT
    # ======================================================

    if "NUM_CTA_HASH" not in dataset.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente "
            "de dataset_final.csv."
        )

    # ======================================================
    # CHARGEMENT DES CLUSTERS
    # ======================================================

    clusters_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / "clients_cluster_test_k5.csv"
    )

    clusters = pd.read_csv(clusters_path)

    print(f"\nFichier clusters : {clusters_path}")
    print(f"Nombre de lignes : {len(clusters)}")

    # ======================================================
    # VÉRIFICATION DU NOMBRE DE LIGNES
    # ======================================================

    if len(dataset) != len(clusters):

        raise ValueError(
            "Le nombre de lignes ne correspond pas : "
            f"{len(dataset)} lignes dans dataset_final.csv "
            f"contre {len(clusters)} lignes dans "
            f"clients_cluster_test_k5.csv."
        )

    # ======================================================
    # VÉRIFICATION DU CLUSTER
    # ======================================================

    if "CLUSTER" not in clusters.columns:

        raise ValueError(
            "La colonne CLUSTER est absente du fichier K-Means."
        )

    # ======================================================
    # VÉRIFICATION DES IDENTIFIANTS
    # ======================================================

    if "NUM_CTA_HASH" not in clusters.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente du fichier "
            "clients_cluster_test_k5.csv."
        )

    # ======================================================
    # VÉRIFICATION DE LA CORRESPONDANCE DES CLIENTS
    # ======================================================

    if not dataset["NUM_CTA_HASH"].equals(
        clusters["NUM_CTA_HASH"]
    ):

        raise ValueError(
            "Les identifiants NUM_CTA_HASH ne correspondent "
            "pas entre les deux fichiers."
        )

    print("\nCorrespondance des clients : OK")

    # ======================================================
    # AJOUT DU CLUSTER
    # ======================================================

    dataset["CLUSTER"] = clusters["CLUSTER"]

    # ======================================================
    # LIBELLÉS MÉTIER
    # ======================================================

    cluster_labels = {

        0: "Faible consommation",

        1: "Consommation moyenne",

        2: "Consommation cumulée élevée - nombreux relevés",

        3: "Très forte consommation",

        4: "Forte consommation"
    }

    dataset["LIBELLE_CLUSTER"] = (
        dataset["CLUSTER"]
        .map(cluster_labels)
    )

    # ======================================================
    # VÉRIFICATION DES LIBELLÉS
    # ======================================================

    if dataset["LIBELLE_CLUSTER"].isna().any():

        raise ValueError(
            "Certains clusters ne possèdent pas "
            "de libellé métier."
        )

    # ======================================================
    # DOSSIER FINAL
    # ======================================================

    output_dir = (
        BASE_DIR
        / "data"
        / "final"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # FICHIER FINAL
    # ======================================================

    output_file = (
        output_dir
        / "clients_segmentes_test.csv"
    )

    dataset.to_csv(
        output_file,
        index=False
    )

    # ======================================================
    # RÉSULTATS
    # ======================================================

    print("\n" + "=" * 60)
    print("SEGMENTATION TERMINÉE")
    print("=" * 60)

    print(
        f"\nNombre total de clients : {len(dataset)}"
    )

    print("\nRépartition des clusters :")

    print(
        dataset["CLUSTER"]
        .value_counts()
        .sort_index()
    )

    print("\nRépartition des profils :")

    print(
        dataset["LIBELLE_CLUSTER"]
        .value_counts()
    )

    print(
        f"\nFichier final enregistré : {output_file}"
    )


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    create_final_dataset_test()