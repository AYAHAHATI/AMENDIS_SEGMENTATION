"""
train_kmeans.py

Entraîne le StandardScaler et le K-Means sur les profils
janvier-avril 2022-2025 (ml/build_training_windows.py).

Sorties :
- models/scaler.pkl
- models/kmeans_model_k{K}.pkl
- models/cluster_labels.json (libellés déduits des centroïdes)
- data/intermediate/clients_cluster_k{K}.csv
- data/intermediate/statistiques_clusters.csv
"""

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.config import (
    INTERMEDIATE_DIR,
    FEATURES,
    MODEL_FEATURES,
    N_CLUSTERS,
    RANDOM_STATE,
    SCALER_FILE,
    KMEANS_FILE,
    ensure_dirs,
)
from ml.labels import build_labels, save_labels

INPUT_FILE = INTERMEDIATE_DIR / "training_windows.csv"


def train_kmeans(k=N_CLUSTERS):
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} introuvable : lancer ml/build_training_windows.py"
        )

    df = pd.read_csv(INPUT_FILE)
    n_total = len(df)
    df = df[df["PROFIL_COMPLET"]].reset_index(drop=True)
    print(
        f"Profils d'entraînement : {len(df):,} complets "
        f"(sur {n_total:,}, {n_total - len(df):,} incomplets exclus)"
    )
    X = df[MODEL_FEATURES]

    # Le scaler est appris sur les profils de 4 mois
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df["CLUSTER"] = model.fit_predict(X_scaled)

    labels, centroids = build_labels(model, scaler)
    df["LIBELLE_CLUSTER"] = df["CLUSTER"].map(labels)

    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(model, KMEANS_FILE)
    save_labels(labels)

    stats = (
        df.groupby(["CLUSTER", "LIBELLE_CLUSTER"])[FEATURES]
        .mean()
        .round(2)
    )
    stats["NB_PROFILS"] = df.groupby(["CLUSTER", "LIBELLE_CLUSTER"]).size()
    stats["PART_%"] = (stats["NB_PROFILS"] / len(df) * 100).round(2)
    stats = stats.sort_values("CONSO_MOYENNE")

    print(f"\nK = {k} | inertie = {model.inertia_:,.2f}")
    print(stats.to_string())

    print("\nRépartition par année (vérifie la stabilité dans le temps) :")
    print(
        pd.crosstab(df["ANNEE"], df["LIBELLE_CLUSTER"], normalize="index")
        .mul(100).round(1).to_string()
    )

    df.to_csv(INTERMEDIATE_DIR / f"clients_cluster_k{k}.csv", index=False)
    stats.to_csv(INTERMEDIATE_DIR / "statistiques_clusters.csv")
    print(f"\nModèles enregistrés : {SCALER_FILE.name}, {KMEANS_FILE.name}")
    return model, scaler, labels


if __name__ == "__main__":
    train_kmeans()
