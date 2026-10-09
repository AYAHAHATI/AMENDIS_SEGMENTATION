import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# DONNÉES
# ==========================================================

features_path = (
    BASE_DIR
    / "data"
    / "intermediate"
    / "scaled_features.csv"
)

features = pd.read_csv(features_path)

variables = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

# ==========================================================
# ANALYSE K = 3
# ==========================================================

clusters_k3 = pd.read_csv(
    BASE_DIR
    / "data"
    / "intermediate"
    / "clients_cluster_k3.csv"
)

data_k3 = features[variables].copy()
data_k3["CLUSTER"] = clusters_k3["CLUSTER"]

stats_k3 = data_k3.groupby("CLUSTER")[variables].mean()

print("\n" + "=" * 70)
print("PROFILS MOYENS — K = 3")
print("=" * 70)
print(stats_k3.round(2))

print("\nNombre de clients par cluster :")
print(
    data_k3["CLUSTER"]
    .value_counts()
    .sort_index()
)

# ==========================================================
# ANALYSE K = 5
# ==========================================================

clusters_k5 = pd.read_csv(
    BASE_DIR
    / "data"
    / "intermediate"
    / "clients_cluster_k5.csv"
)

data_k5 = features[variables].copy()
data_k5["CLUSTER"] = clusters_k5["CLUSTER"]

stats_k5 = data_k5.groupby("CLUSTER")[variables].mean()

print("\n" + "=" * 70)
print("PROFILS MOYENS — K = 5")
print("=" * 70)
print(stats_k5.round(2))

print("\nNombre de clients par cluster :")
print(
    data_k5["CLUSTER"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)