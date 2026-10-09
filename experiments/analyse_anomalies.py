"""
analyse_anomalies.py

Analyse des anomalies détectées par Isolation Forest
sur les données historiques 2022-2025.
"""

import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = "/opt/airflow/data/final/anomaly_scores_historical.csv"

FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]


# ==========================================================
# CHARGEMENT
# ==========================================================

print("=" * 70)
print("ANALYSE DES ANOMALIES - ISOLATION FOREST")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nNombre total de clients : {len(df)}")

print("\nColonnes disponibles :")
print(list(df.columns))


# ==========================================================
# SÉLECTION DES ANOMALIES
# ==========================================================

df_anomalies = df[
    df["ANOMALIE_ISOLATION_FOREST"] == True
].copy()

print("\n" + "=" * 70)
print("RÉSULTATS")
print("=" * 70)

print(
    f"\nNombre d'anomalies : {len(df_anomalies)}"
)

print(
    f"Pourcentage : "
    f"{len(df_anomalies) / len(df) * 100:.2f}%"
)


# ==========================================================
# STATISTIQUES DES ANOMALIES
# ==========================================================

print("\n" + "=" * 70)
print("PROFIL STATISTIQUE DES ANOMALIES")
print("=" * 70)

print(
    df_anomalies[FEATURES]
    .describe()
    .round(2)
    .to_string()
)


# ==========================================================
# COMPARAISON ANOMALIES / CLIENTS NORMAUX
# ==========================================================

df_normaux = df[
    df["ANOMALIE_ISOLATION_FOREST"] == False
].copy()

print("\n" + "=" * 70)
print("COMPARAISON ANOMALIES / CLIENTS NORMAUX")
print("=" * 70)

comparaison = pd.DataFrame({
    "Anomalies": df_anomalies[FEATURES].mean(),
    "Clients_normaux": df_normaux[FEATURES].mean()
})

comparaison["Ratio_anomalie_normal"] = (
    comparaison["Anomalies"]
    / comparaison["Clients_normaux"]
)

print(
    comparaison.round(2).to_string()
)


# ==========================================================
# ANOMALIES LES PLUS EXTRÊMES
# ==========================================================

print("\n" + "=" * 70)
print("ANOMALIES LES PLUS EXTRÊMES")
print("=" * 70)


for feature in FEATURES:

    print(f"\n--- {feature} ---")

    top = (
        df_anomalies
        .sort_values(feature, ascending=False)
        [["NUM_CTA_HASH", feature]]
        .head(10)
    )

    print(top.to_string(index=False))


# ==========================================================
# SAUVEGARDE
# ==========================================================

OUTPUT_FILE = (
    "/opt/airflow/data/final/anomalies_analysis.csv"
)

comparaison.to_csv(
    OUTPUT_FILE,
    index=True
)

print("\n" + "=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)

print(f"\nFichier généré :")
print(OUTPUT_FILE)