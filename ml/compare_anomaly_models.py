"""
compare_anomaly_models.py

Comparaison de trois modèles de détection d'anomalies
sur les données historiques 2022-2025 :

1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM

Les mêmes variables et les mêmes données sont utilisées
pour les trois modèles afin d'obtenir une comparaison
équitable.
"""

import time

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ==========================================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/clients_segmentes.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("COMPARAISON DES MODÈLES DE DÉTECTION D'ANOMALIES")
print("=" * 70)

print(f"Nombre de clients : {len(df)}")


# ==========================================================
# 2. SÉLECTION DES FEATURES
# ==========================================================

FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

X = df[FEATURES].copy()

print("\nFeatures utilisées :")
print(FEATURES)


# ==========================================================
# 3. STANDARDISATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardisation terminée.")


# ==========================================================
# 4. PARAMÈTRE COMMUN
# ==========================================================

# Pour comparer les modèles dans les mêmes conditions,
# nous considérons ici environ 5 % des observations comme
# anomalies potentielles.

CONTAMINATION = 0.05


# ==========================================================
# 5. ISOLATION FOREST
# ==========================================================

print("\n" + "=" * 70)
print("1. ISOLATION FOREST")
print("=" * 70)

start = time.time()

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42
)

isolation_labels = isolation_forest.fit_predict(X_scaled)

isolation_time = time.time() - start

# -1 = anomalie
#  1 = normal
isolation_anomalies = np.sum(isolation_labels == -1)

# Score élevé = plus anormal
isolation_scores = -isolation_forest.decision_function(X_scaled)

print(f"Anomalies détectées : {isolation_anomalies}")
print(
    f"Pourcentage d'anomalies : "
    f"{isolation_anomalies / len(X) * 100:.2f}%"
)
print(f"Temps d'exécution : {isolation_time:.4f} secondes")


# ==========================================================
# 6. LOCAL OUTLIER FACTOR
# ==========================================================

print("\n" + "=" * 70)
print("2. LOCAL OUTLIER FACTOR (LOF)")
print("=" * 70)

start = time.time()

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION
)

lof_labels = lof.fit_predict(X_scaled)

lof_time = time.time() - start

# -1 = anomalie
#  1 = normal
lof_anomalies = np.sum(lof_labels == -1)

# Score élevé = plus anormal
lof_scores = -lof.negative_outlier_factor_

print(f"Anomalies détectées : {lof_anomalies}")
print(
    f"Pourcentage d'anomalies : "
    f"{lof_anomalies / len(X) * 100:.2f}%"
)
print(f"Temps d'exécution : {lof_time:.4f} secondes")


# ==========================================================
# 7. ONE-CLASS SVM
# ==========================================================

print("\n" + "=" * 70)
print("3. ONE-CLASS SVM")
print("=" * 70)

start = time.time()

one_class_svm = OneClassSVM(
    kernel="rbf",
    nu=CONTAMINATION,
    gamma="scale"
)

svm_labels = one_class_svm.fit_predict(X_scaled)

svm_time = time.time() - start

# -1 = anomalie
#  1 = normal
svm_anomalies = np.sum(svm_labels == -1)

# Score élevé = plus anormal
svm_scores = -one_class_svm.decision_function(X_scaled)

print(f"Anomalies détectées : {svm_anomalies}")
print(
    f"Pourcentage d'anomalies : "
    f"{svm_anomalies / len(X) * 100:.2f}%"
)
print(f"Temps d'exécution : {svm_time:.4f} secondes")


# ==========================================================
# 8. TABLEAU DE COMPARAISON
# ==========================================================

comparison = pd.DataFrame({
    "MODELE": [
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ],
    "NB_ANOMALIES": [
        isolation_anomalies,
        lof_anomalies,
        svm_anomalies
    ],
    "POURCENTAGE_ANOMALIES": [
        isolation_anomalies / len(X) * 100,
        lof_anomalies / len(X) * 100,
        svm_anomalies / len(X) * 100
    ],
    "TEMPS_EXECUTION_SEC": [
        isolation_time,
        lof_time,
        svm_time
    ]
})


# ==========================================================
# 9. AFFICHAGE DES RÉSULTATS
# ==========================================================

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# 10. SAUVEGARDE DES SCORES
# ==========================================================

results = df.copy()

results["ANOMALIE_ISOLATION_FOREST"] = (
    isolation_labels == -1
)

results["SCORE_ISOLATION_FOREST"] = (
    isolation_scores
)

results["ANOMALIE_LOF"] = (
    lof_labels == -1
)

results["SCORE_LOF"] = (
    lof_scores
)

results["ANOMALIE_ONE_CLASS_SVM"] = (
    svm_labels == -1
)

results["SCORE_ONE_CLASS_SVM"] = (
    svm_scores
)


# ==========================================================
# 11. SAUVEGARDE
# ==========================================================

OUTPUT_COMPARISON = (
    "/opt/airflow/data/final/"
    "anomaly_models_comparison.csv"
)

OUTPUT_SCORES = (
    "/opt/airflow/data/final/"
    "anomaly_scores_historical.csv"
)

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

results.to_csv(
    OUTPUT_SCORES,
    index=False
)

print("\nFichiers générés :")
print(OUTPUT_COMPARISON)
print(OUTPUT_SCORES)

print("\nComparaison terminée.")