"""
create_final_dataset.py

Création du fichier final des clients segmentés 2026.
"""

import pandas as pd
from pathlib import Path


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# CRÉATION DU DATASET FINAL
# ==========================================================

def create_final_dataset(
    source_file="hist_2026_client.csv",
    clusters_file="clients_cluster_2026.csv",
    output_file="clients_segmentes_2026.csv"
):

    print("\n" + "=" * 60)
    print("CRÉATION DU DATASET FINAL DE SEGMENTATION 2026")
    print("=" * 60)

    # ======================================================
    # CHEMINS
    # ======================================================

    intermediate_dir = (
        BASE_DIR
        / "data"
        / "intermediate"
    )

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
    # CHARGEMENT DU DATASET CLIENTS 2026
    # ======================================================

    dataset_path = (
        intermediate_dir
        / source_file
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset source introuvable : {dataset_path}"
        )

    dataset = pd.read_csv(
        dataset_path
    )

    print(
        f"\nDataset source : {dataset_path}"
    )

    print(
        f"Nombre de lignes : {len(dataset)}"
    )

    # ======================================================
    # VÉRIFICATION DE L'IDENTIFIANT SOURCE
    # ======================================================

    if "NUM_CTA_HASH" not in dataset.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente "
            "du dataset source."
        )

    # ======================================================
    # VÉRIFICATION DES IDENTIFIANTS SOURCE
    # ======================================================

    print(
        f"\nIdentifiants uniques source : "
        f"{dataset['NUM_CTA_HASH'].nunique()}"
    )

    if dataset["NUM_CTA_HASH"].duplicated().any():

        duplicates = (
            dataset["NUM_CTA_HASH"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Le dataset source contient "
            f"{duplicates} identifiants dupliqués."
        )

    # ======================================================
    # CHARGEMENT DES CLUSTERS 2026
    # ======================================================

    clusters_path = (
        intermediate_dir
        / clusters_file
    )

    if not clusters_path.exists():

        raise FileNotFoundError(
            f"Fichier clusters introuvable : "
            f"{clusters_path}"
        )

    clusters = pd.read_csv(
        clusters_path
    )

    print(
        f"\nFichier clusters : {clusters_path}"
    )

    print(
        f"Nombre de lignes : {len(clusters)}"
    )

    # ======================================================
    # VÉRIFICATION DES COLONNES
    # ======================================================

    if "NUM_CTA_HASH" not in clusters.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente "
            "du fichier clusters."
        )

    if "CLUSTER" not in clusters.columns:

        raise ValueError(
            "La colonne CLUSTER est absente "
            "du fichier clusters."
        )

    # ======================================================
    # VÉRIFICATION DES IDENTIFIANTS CLUSTERS
    # ======================================================

    print(
        f"\nIdentifiants uniques clusters : "
        f"{clusters['NUM_CTA_HASH'].nunique()}"
    )

    if clusters["NUM_CTA_HASH"].duplicated().any():

        duplicates = (
            clusters["NUM_CTA_HASH"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Le fichier clusters contient "
            f"{duplicates} identifiants dupliqués."
        )

    # ======================================================
    # VÉRIFICATION DU NOMBRE DE CLIENTS
    # ======================================================

    if len(dataset) != len(clusters):

        raise ValueError(
            "\nERREUR : le nombre de clients ne correspond pas.\n"
            f"Dataset 2026 : {len(dataset)}\n"
            f"Clusters 2026 : {len(clusters)}"
        )

    print(
        "\nNombre de clients : OK"
    )

    # ======================================================
    # VÉRIFICATION DES IDENTIFIANTS
    # ======================================================

    source_ids = set(
        dataset["NUM_CTA_HASH"]
    )

    cluster_ids = set(
        clusters["NUM_CTA_HASH"]
    )

    ids_manquants = (
        source_ids - cluster_ids
    )

    ids_supplementaires = (
        cluster_ids - source_ids
    )

    if ids_manquants:

        raise ValueError(
            f"{len(ids_manquants)} clients du dataset "
            "source n'ont pas de cluster."
        )

    if ids_supplementaires:

        raise ValueError(
            f"{len(ids_supplementaires)} identifiants "
            "clusters ne sont pas présents dans le dataset source."
        )

    print(
        "Correspondance des identifiants NUM_CTA_HASH : OK"
    )

    # ======================================================
    # FUSION DES DONNÉES
    # ======================================================
    #
    # IMPORTANT :
    # On fait la correspondance grâce à NUM_CTA_HASH
    # et non grâce à la position des lignes.
    #
    # Cela rend le pipeline beaucoup plus robuste.
    #

    clusters_selection = clusters[
        [
            "NUM_CTA_HASH",
            "CLUSTER"
        ]
    ].copy()

    dataset_final = dataset.merge(
        clusters_selection,
        on="NUM_CTA_HASH",
        how="left",
        validate="one_to_one"
    )

    # ======================================================
    # VÉRIFICATION APRÈS FUSION
    # ======================================================

    if len(dataset_final) != len(dataset):

        raise ValueError(
            "\nERREUR après fusion : "
            "le nombre de lignes a changé.\n"
            f"Avant : {len(dataset)}\n"
            f"Après : {len(dataset_final)}"
        )

    if dataset_final["CLUSTER"].isna().any():

        nombre_sans_cluster = (
            dataset_final["CLUSTER"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{nombre_sans_cluster} clients "
            "n'ont pas reçu de cluster."
        )

    print(
        "\nFusion des données : OK"
    )

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

    dataset_final["LIBELLE_CLUSTER"] = (
        dataset_final["CLUSTER"]
        .map(cluster_labels)
    )

    # ======================================================
    # VÉRIFICATION DES LIBELLÉS
    # ======================================================

    if dataset_final["LIBELLE_CLUSTER"].isna().any():

        clusters_inconnus = (
            dataset_final.loc[
                dataset_final["LIBELLE_CLUSTER"].isna(),
                "CLUSTER"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Clusters sans libellé métier : "
            f"{clusters_inconnus}"
        )

    print(
        "Libellés métier : OK"
    )

    # ======================================================
    # SAUVEGARDE
    # ======================================================

    output_path = (
        output_dir
        / output_file
    )

    dataset_final.to_csv(
        output_path,
        index=False
    )

    # ======================================================
    # RÉSULTATS
    # ======================================================

    print("\n" + "=" * 60)
    print("SEGMENTATION 2026 TERMINÉE")
    print("=" * 60)

    print(
        f"\nNombre total de clients : "
        f"{len(dataset_final)}"
    )

    print(
        f"Nombre d'identifiants uniques : "
        f"{dataset_final['NUM_CTA_HASH'].nunique()}"
    )

    print(
        "\nRépartition des clusters :"
    )

    print(
        dataset_final["CLUSTER"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nRépartition des profils :"
    )

    print(
        dataset_final["LIBELLE_CLUSTER"]
        .value_counts()
    )

    print(
        f"\nFichier final enregistré : "
        f"{output_path}"
    )

    return dataset_final


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    create_final_dataset()