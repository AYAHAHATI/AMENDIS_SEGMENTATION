"""
test_stability_isolation_forest.py

Test de stabilité du modèle Isolation Forest
sur les données historiques 2022-2025.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/clients_segmentes.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("TEST DE STABILITÉ - ISOLATION FOREST")
print("=" * 70)

print(f"Nombre de clients : {len(df)}")


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

X = df[FEATURES].copy()


# ==========================================================
# 3. STANDARDISATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardisation terminée.")


# ==========================================================
# 4. TEST AVEC PLUSIEURS CONTAMINATIONS
# ==========================================================

contaminations = [
    0.01,
    0.03,
    0.05,
    0.10
]

results = []


for contamination in contaminations:

    print("\n" + "-" * 70)

    print(
        f"Isolation Forest avec contamination = "
        f"{contamination * 100:.0f}%"
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    anomaly_count = np.sum(labels == -1)

    print(
        f"Nombre d'anomalies : "
        f"{anomaly_count}"
    )

    print(
        f"Pourcentage réel : "
        f"{anomaly_count / len(df) * 100:.2f}%"
    )

    results.append({
        "CONTAMINATION": contamination,
        "NB_ANOMALIES": anomaly_count,
        "POURCENTAGE": anomaly_count / len(df) * 100
    })


# ==========================================================
# 5. TABLEAU FINAL
# ==========================================================

comparison = pd.DataFrame(results)

print("\n" + "=" * 70)
print("RÉSULTAT DU TEST DE STABILITÉ")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# 6. SAUVEGARDE
# ==========================================================

OUTPUT_FILE = (
    "/opt/airflow/data/final/"
    "isolation_forest_stability.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFichier généré :")
print(OUTPUT_FILE)

print("\nTest de stabilité terminé.")