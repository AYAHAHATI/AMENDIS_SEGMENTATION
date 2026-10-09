"""
test_stability_clients.py

Vérification de la stabilité des clients détectés
par Isolation Forest lorsque la contamination varie.

Données historiques : 2022-2025
"""

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CHARGEMENT
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/clients_segmentes.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STABILITÉ DES CLIENTS DÉTECTÉS - ISOLATION FOREST")
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
# 4. DÉTECTION POUR CHAQUE CONTAMINATION
# ==========================================================

contaminations = [0.01, 0.03, 0.05, 0.10]

anomaly_sets = {}

for contamination in contaminations:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    # On utilise l'identifiant client
    clients = set(
        df.loc[
            labels == -1,
            "NUM_CTA_HASH"
        ]
    )

    anomaly_sets[contamination] = clients

    print(
        f"\nContamination {contamination * 100:.0f}% "
        f"→ {len(clients)} anomalies"
    )


# ==========================================================
# 5. COMPARAISON ENTRE LES NIVEAUX
# ==========================================================

print("\n" + "=" * 70)
print("STABILITÉ ENTRE LES NIVEAUX")
print("=" * 70)


pairs = [
    (0.01, 0.03),
    (0.03, 0.05),
    (0.05, 0.10)
]


for c1, c2 in pairs:

    set1 = anomaly_sets[c1]
    set2 = anomaly_sets[c2]

    common = set1 & set2

    # Pourcentage des anomalies du premier niveau
    # retrouvées au niveau suivant
    stability = (
        len(common) / len(set1) * 100
    )

    print(
        f"\n{c1 * 100:.0f}% → {c2 * 100:.0f}%"
    )

    print(
        f"Clients communs : {len(common)}"
    )

    print(
        f"Stabilité : {stability:.2f}%"
    )


# ==========================================================
# 6. STABILITÉ DES 1 % LES PLUS ATYPIQUES
# ==========================================================

print("\n" + "=" * 70)
print("STABILITÉ DES ANOMALIES LES PLUS EXTRÊMES")
print("=" * 70)


base = anomaly_sets[0.01]

for contamination in [0.03, 0.05, 0.10]:

    current = anomaly_sets[contamination]

    common = base & current

    stability = (
        len(common) / len(base) * 100
    )

    print(
        f"\n1% → {contamination * 100:.0f}%"
    )

    print(
        f"Anomalies communes : {len(common)} / {len(base)}"
    )

    print(
        f"Stabilité : {stability:.2f}%"
    )


print("\n" + "=" * 70)
print("TEST TERMINÉ")
print("=" * 70)