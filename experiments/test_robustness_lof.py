"""
test_robustness_lof.py

Test de sensibilité du modèle Local Outlier Factor (LOF)
à différents nombres de voisins.

Données historiques : 2022-2025
"""

import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/clients_segmentes.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("TEST DE ROBUSTESSE - LOCAL OUTLIER FACTOR")
print("=" * 70)

print(f"Nombre total de clients : {len(df)}")


# ==========================================================
# 2. FEATURES
# ==========================================================

FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

X = df[FEATURES]


# ==========================================================
# 3. STANDARDISATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardisation terminée.")


# ==========================================================
# 4. TEST DE PLUSIEURS NOMBRE DE VOISINS
# ==========================================================

neighbors_values = [10, 20, 30, 50]

contamination = 0.05

anomaly_sets = {}


for n_neighbors in neighbors_values:

    print("\n" + "-" * 70)

    print(
        f"LOF avec n_neighbors = {n_neighbors}"
    )

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    labels = model.fit_predict(X_scaled)

    clients = set(
        df.loc[
            labels == -1,
            "NUM_CTA_HASH"
        ]
    )

    anomaly_sets[n_neighbors] = clients

    print(
        f"Nombre d'anomalies : {len(clients)}"
    )


# ==========================================================
# 5. MODÈLE DE RÉFÉRENCE
# ==========================================================

reference_neighbors = 20

reference_set = anomaly_sets[reference_neighbors]


# ==========================================================
# 6. STABILITÉ PAR RAPPORT À n_neighbors = 20
# ==========================================================

print("\n" + "=" * 70)
print("STABILITÉ PAR RAPPORT À n_neighbors = 20")
print("=" * 70)


results = []


for n_neighbors in neighbors_values:

    current_set = anomaly_sets[n_neighbors]

    common = reference_set & current_set

    stability = (
        len(common)
        / len(reference_set)
        * 100
    )

    results.append({
        "N_NEIGHBORS": n_neighbors,
        "NB_ANOMALIES": len(current_set),
        "ANOMALIES_COMMUNES": len(common),
        "STABILITE_POURCENT": stability
    })

    print(
        f"\nn_neighbors = {n_neighbors}"
    )

    print(
        f"Anomalies communes : "
        f"{len(common)} / {len(reference_set)}"
    )

    print(
        f"Stabilité : {stability:.2f}%"
    )


# ==========================================================
# 7. TABLEAU FINAL
# ==========================================================

comparison = pd.DataFrame(results)

print("\n" + "=" * 70)
print("TABLEAU DE ROBUSTESSE LOF")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# 8. SAUVEGARDE
# ==========================================================

OUTPUT_FILE = (
    "/opt/airflow/data/final/"
    "lof_robustness.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFichier généré :")
print(OUTPUT_FILE)

print("\nTest LOF terminé.")