import pandas as pd

# Données utilisées pour la segmentation
features = pd.read_csv("data/intermediate/features.csv")

# Résultat du K-Means
clusters = pd.read_csv("data/intermediate/clients_cluster_k5.csv")

# Ajouter la colonne CLUSTER
features["CLUSTER"] = clusters["CLUSTER"]

# Statistiques de consommation par cluster
stats = features.groupby("CLUSTER").agg({
    "CONSO_TOTALE": "mean",
    "CONSO_MOYENNE": "mean",
    "CONSO_MAX": "mean",
    "CONSO_MIN": "mean",
    "NB_RELEVES": "mean"
})

print("\n===== STATISTIQUES PAR CLUSTER =====\n")
print(stats.round(2))

# Nombre de clients par cluster
print("\n===== NOMBRE DE CLIENTS PAR CLUSTER =====\n")
print(features["CLUSTER"].value_counts().sort_index())

# Sauvegarde
stats.to_csv("data/intermediate/statistiques_clusters.csv")

print("\nFichier enregistré : data/intermediate/statistiques_clusters.csv")