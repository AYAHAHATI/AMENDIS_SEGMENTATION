import pandas as pd

# Données originales (avant standardisation)
features = pd.read_csv("data/intermediate/features.csv")

# Résultat du KMeans
clusters = pd.read_csv("data/intermediate/clients_cluster_k5.csv")

# Ajouter la colonne CLUSTER
features["CLUSTER"] = clusters["CLUSTER"]

# Statistiques par cluster
stats = features.groupby("CLUSTER").agg({
    "CONSO_TOTALE": "mean",
    "CONSO_MOYENNE": "mean",
    "CONSO_MAX": "mean",
    "CONSO_MIN": "mean",
    "MONTANT_TOTAL": "mean",
    "MONTANT_MOYEN": "mean",
    "MONTANT_MAX": "mean",
    "NB_FACTURES": "mean",
    "VOLUME_FACTURE": "mean"
})

print("\n===== STATISTIQUES PAR CLUSTER =====\n")
print(stats.round(2))

# Sauvegarde
stats.to_csv("data/intermediate/statistiques_clusters.csv")

print("\nFichier enregistré : data/intermediate/statistiques_clusters.csv")