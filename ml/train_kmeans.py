"""
train_kmeans.py

Entraînement de modèles K-Means pour la segmentation
des clients Amendis à partir des variables de consommation.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans


# ==========================================================
# RACINE DU PROJET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================================
# FONCTION PRINCIPALE
# ==========================================================

def train_kmeans(
    input_file="scaled_features.csv",
    dataset_file="dataset_final.csv",
    output_prefix="clients_cluster",
    model_prefix="kmeans_model"
):

    print("\n" + "=" * 60)
    print("CHARGEMENT DES DONNÉES POUR K-MEANS")
    print("=" * 60)

    # ======================================================
    # CHARGEMENT DES FEATURES STANDARDISÉES
    # ======================================================

    features_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / input_file
    )

    if not features_path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {features_path}"
        )

    features = pd.read_csv(features_path)

    print(
        f"\nFichier utilisé : {features_path}"
    )

    print(
        f"Nombre de clients : {len(features)}"
    )

    print(
        f"Nombre de variables : {features.shape[1]}"
    )

    # ======================================================
    # VARIABLES UTILISÉES POUR K-MEANS
    # ======================================================

    expected_columns = [
        "CONSO_TOTALE",
        "CONSO_MOYENNE",
        "CONSO_MAX",
        "CONSO_MIN",
        "NB_RELEVES"
    ]

    # ======================================================
    # VÉRIFICATION DES COLONNES
    # ======================================================

    missing_columns = [
        col
        for col in expected_columns
        if col not in features.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Variables manquantes dans {input_file} : "
            f"{missing_columns}"
        )

    # ======================================================
    # CONSERVATION DES VARIABLES DE CONSOMMATION
    # ======================================================

    features = features[expected_columns]

    print("\nVariables utilisées :")
    print(expected_columns)

    # ======================================================
    # CHARGEMENT DU DATASET CLIENTS
    # ======================================================

    dataset_path = (
        BASE_DIR
        / "data"
        / "intermediate"
        / dataset_file
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {dataset_path}"
        )

    dataset = pd.read_csv(dataset_path)

    print(
        f"\nDataset clients : {dataset_path}"
    )

    print(
        f"Nombre de lignes : {len(dataset)}"
    )

    # ======================================================
    # VÉRIFICATION DE L'IDENTIFIANT
    # ======================================================

    if "NUM_CTA_HASH" not in dataset.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente "
            f"de {dataset_file}."
        )

    # ======================================================
    # VÉRIFICATION DU NOMBRE DE LIGNES
    # ======================================================

    if len(features) != len(dataset):

        raise ValueError(
            "\nLe nombre de lignes ne correspond pas :\n"
            f"Features : {len(features)}\n"
            f"Clients : {len(dataset)}\n"
            f"Features utilisées : {input_file}\n"
            f"Dataset utilisé : {dataset_file}"
        )

    # ======================================================
    # IDENTIFIANTS CLIENTS
    # ======================================================

    client_ids = dataset["NUM_CTA_HASH"].copy()

    print(
        f"\nNombre d'identifiants clients : "
        f"{len(client_ids)}"
    )

    # ======================================================
    # DOSSIER DES MODÈLES
    # ======================================================

    models_dir = BASE_DIR / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # ENTRAÎNEMENT K-MEANS
    # ======================================================

    for k in [2, 5]:

        print("\n" + "=" * 60)
        print(f"K-MEANS : K = {k}")
        print("=" * 60)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        # ==================================================
        # ENTRAÎNEMENT
        # ==================================================

        labels = model.fit_predict(features)

        # ==================================================
        # CRÉATION DU RÉSULTAT
        # ==================================================

        result = features.copy()

        result.insert(
            0,
            "NUM_CTA_HASH",
            client_ids.values
        )

        result["CLUSTER"] = labels

        # ==================================================
        # AFFICHAGE
        # ==================================================

        print("\nNombre de clients par cluster :")

        print(
            result["CLUSTER"]
            .value_counts()
            .sort_index()
        )

        print(
            f"\nInertie : {model.inertia_}"
        )

        # ==================================================
        # SAUVEGARDE DU RÉSULTAT
        # ==================================================

        output_file = (
            BASE_DIR
            / "data"
            / "intermediate"
            / f"{output_prefix}_k{k}.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nFichier enregistré : {output_file}"
        )

        # ==================================================
        # SAUVEGARDE DU MODÈLE
        # ==================================================

        model_path = (
            models_dir
            / f"{model_prefix}_k{k}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Modèle enregistré : {model_path}"
        )

    print("\n" + "=" * 60)
    print("K-MEANS TERMINÉ")
    print("=" * 60)


# ==========================================================
# EXÉCUTION DIRECTE
# ==========================================================

if __name__ == "__main__":

    train_kmeans()