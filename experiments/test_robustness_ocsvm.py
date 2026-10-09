"""
test_robustness_ocsvm.py

Test de sensibilité du modèle One-Class SVM
à différentes valeurs du paramètre nu.

Données historiques : 2022-2025
"""

import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/clients_segmentes.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("TEST DE ROBUSTESSE - ONE-CLASS SVM")
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
# 4. TEST DE PLUSIEURS VALEURS DE NU
# ==========================================================

nu_values = [0.03, 0.05, 0.07, 0.10]

anomaly_sets = {}


for nu in nu_values:

    print("\n" + "-" * 70)

    print(
        f"One-Class SVM avec nu = {nu}"
    )

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu
    )

    labels = model.fit_predict(X_scaled)

    clients = set(
        df.loc[
            labels == -1,
            "NUM_CTA_HASH"
        ]
    )

    anomaly_sets[nu] = clients

    print(
        f"Nombre d'anomalies : {len(clients)}"
    )

    print(
        f"Pourcentage : "
        f"{len(clients) / len(df) * 100:.2f}%"
    )


# ==========================================================
# 5. MODÈLE DE RÉFÉRENCE
# ==========================================================

reference_nu = 0.05

reference_set = anomaly_sets[reference_nu]


# ==========================================================
# 6. TEST DE STABILITÉ
# ==========================================================

print("\n" + "=" * 70)
print("STABILITÉ PAR RAPPORT À nu = 0.05")
print("=" * 70)


results = []


for nu in nu_values:

    current_set = anomaly_sets[nu]

    common = reference_set & current_set

    stability = (
        len(common)
        / len(reference_set)
        * 100
    )

    results.append({
        "NU": nu,
        "NB_ANOMALIES": len(current_set),
        "ANOMALIES_COMMUNES": len(common),
        "STABILITE_POURCENT": stability
    })

    print(
        f"\nnu = {nu}"
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
print("TABLEAU DE ROBUSTESSE ONE-CLASS SVM")
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
    "one_class_svm_robustness.csv"
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFichier généré :")
print(OUTPUT_FILE)

print("\nTest One-Class SVM terminé.")