"""
train_kmeans_k3.py

Test du modèle K-Means avec K = 3
pour comparaison avec les configurations K = 2 et K = 5.
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
# CHARGEMENT DES FEATURES STANDARDISÉES
# ==========================================================

features_path = (
    BASE_DIR
    / "data"
    / "intermediate"
    / "scaled_features.csv"
)

features = pd.read_csv(features_path)

expected_columns = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

features = features[expected_columns]


print("\n" + "=" * 60)
print("K-MEANS : TEST AVEC K = 3")
print("=" * 60)

print(f"\nNombre de clients : {len(features)}")

print("\nVariables utilisées :")
print(expected_columns)


# ==========================================================
# ENTRAÎNEMENT
# ==========================================================

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(features)


# ==========================================================
# RÉSULTATS
# ==========================================================

result = features.copy()
result["CLUSTER"] = labels

print("\nNombre de clients par cluster :")

print(
    result["CLUSTER"]
    .value_counts()
    .sort_index()
)

print(
    f"\nInertie : {model.inertia_}"
)


# ==========================================================
# SAUVEGARDE DU RÉSULTAT
# ==========================================================

output_file = (
    BASE_DIR
    / "data"
    / "intermediate"
    / "clients_cluster_k3.csv"
)

result.to_csv(
    output_file,
    index=False
)

print(
    f"\nFichier enregistré : {output_file}"
)


# ==========================================================
# SAUVEGARDE DU MODÈLE
# ==========================================================

models_dir = BASE_DIR / "models"

models_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_path = (
    models_dir
    / "kmeans_model_k3.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    f"Modèle enregistré : {model_path}"
)


print("\n" + "=" * 60)
print("TEST K = 3 TERMINÉ")
print("=" * 60)