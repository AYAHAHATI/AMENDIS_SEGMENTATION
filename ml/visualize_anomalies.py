"""
visualize_anomalies.py

Visualisation des anomalies détectées par Isolation Forest
sur les données historiques 2022-2025.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = (
    "/opt/airflow/data/final/"
    "anomaly_scores_historical.csv"
)

OUTPUT_DIR = (
    "/opt/airflow/data/final/"
    "anomaly_visualizations"
)

FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN"
]


# ==========================================================
# CRÉATION DU DOSSIER
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

print("=" * 70)
print("VISUALISATION DES ANOMALIES - ISOLATION FOREST")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(
    f"\nNombre total de clients : {len(df)}"
)


# ==========================================================
# SÉPARATION DES CLIENTS
# ==========================================================

df_anomalies = df[
    df["ANOMALIE_ISOLATION_FOREST"] == True
].copy()

df_normaux = df[
    df["ANOMALIE_ISOLATION_FOREST"] == False
].copy()

print(
    f"Nombre d'anomalies : {len(df_anomalies)}"
)

print(
    f"Nombre de clients normaux : {len(df_normaux)}"
)


# ==========================================================
# GRAPHIQUE 1 : CONSOMMATION TOTALE
# ==========================================================

print("\nCréation du graphique 1...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df_normaux["CONSO_TOTALE"],
        df_anomalies["CONSO_TOTALE"]
    ],
    labels=[
        "Clients normaux",
        "Anomalies"
    ]
)

plt.title(
    "Distribution de la consommation totale"
)

plt.ylabel(
    "Consommation totale"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_conso_totale.png"
    ),
    dpi=150
)

plt.close()


# ==========================================================
# GRAPHIQUE 2 : CONSOMMATION MOYENNE
# ==========================================================

print("Création du graphique 2...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df_normaux["CONSO_MOYENNE"],
        df_anomalies["CONSO_MOYENNE"]
    ],
    labels=[
        "Clients normaux",
        "Anomalies"
    ]
)

plt.title(
    "Distribution de la consommation moyenne"
)

plt.ylabel(
    "Consommation moyenne"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_conso_moyenne.png"
    ),
    dpi=150
)

plt.close()


# ==========================================================
# GRAPHIQUE 3 : CONSOMMATION MAXIMALE
# ==========================================================

print("Création du graphique 3...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df_normaux["CONSO_MAX"],
        df_anomalies["CONSO_MAX"]
    ],
    labels=[
        "Clients normaux",
        "Anomalies"
    ]
)

plt.title(
    "Distribution de la consommation maximale"
)

plt.ylabel(
    "Consommation maximale"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_conso_max.png"
    ),
    dpi=150
)

plt.close()


# ==========================================================
# GRAPHIQUE 4 : NOMBRE DE RELEVÉS
# ==========================================================

print("Création du graphique 4...")

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df_normaux["NB_RELEVES"],
        df_anomalies["NB_RELEVES"]
    ],
    labels=[
        "Clients normaux",
        "Anomalies"
    ]
)

plt.title(
    "Distribution du nombre de relevés"
)

plt.ylabel(
    "Nombre de relevés"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_nb_releves.png"
    ),
    dpi=150
)

plt.close()


# ==========================================================
# RÉSUMÉ
# ==========================================================

print("\n" + "=" * 70)
print("VISUALISATIONS TERMINÉES")
print("=" * 70)

print(
    f"\nLes graphiques sont enregistrés dans :"
)

print(OUTPUT_DIR)

print("\nFichiers générés :")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(
        f"- {filename}"
    )

print("\nVisualisation terminée.")