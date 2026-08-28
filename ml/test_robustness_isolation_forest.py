"""
test_robustness_isolation_forest.py

Test de robustesse d'Isolation Forest avec
plusieurs initialisations aléatoires.

Données historiques : 2022-2025
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
print("TEST DE ROBUSTESSE - ISOLATION FOREST")
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


# ==========================================================
# 4. DIFFÉRENTS RANDOM_STATE
# ==========================================================

random_states = [10, 20, 30, 40, 42]

contamination = 0.05

anomaly_sets = {}


for random_state in random_states:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )

    labels = model.fit_predict(X_scaled)

    clients = set(
        df.loc[
            labels == -1,
            "NUM_CTA_HASH"
        ]
    )

    anomaly_sets[random_state] = clients

    print(
        f"\nRandom state {random_state}"
    )

    print(
        f"Nombre d'anomalies : {len(clients)}"
    )


# ==========================================================
# 5. COMPARAISON AVEC LE MODÈLE DE RÉFÉRENCE
# ==========================================================

reference_state = 42

reference_set = anomaly_sets[reference_state]

print("\n" + "=" * 70)
print("STABILITÉ PAR RAPPORT AU MODÈLE DE RÉFÉRENCE")
print("=" * 70)

results = []


for random_state in random_states:

    current_set = anomaly_sets[random_state]

    common = reference_set & current_set

    stability = (
        len(common)
        / len(reference_set)
        * 100
    )

    results.append({
        "RANDOM_STATE": random_state,
        "NB_ANOMALIES": len(current_set),
        "ANOMALIES_COMMUNES": len(common),
        "STABILITE_POURCENT": stability
    })

    print(
        f"\nRandom state {random_state}"
    )

    print(
        f"Anomalies communes avec 42 : "
        f"{len(common)} / {len(reference_set)}"
    )

    print(
        f"Stabilité : {stability:.2f}%"
    )


# ==========================================================
# 6. TABLEAU FINAL
# ==========================================================

comparison = pd.DataFrame(results)

print("\n" + "=" * 70)
print("TABLEAU DE ROBUSTESSE")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# 7. SAUVEGARDE
# ==========================================================

OUTPUT_FILE = (
    "/opt/airflow/data/final/"
    "isolation_forest_robustness.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFichier généré :")
print(OUTPUT_FILE)

print("\nTest de robustesse terminé.")