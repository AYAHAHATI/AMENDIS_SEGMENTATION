"""
choose_k.py

Choix du meilleur nombre de clusters (K)
avec :
- la méthode du coude (Elbow Method)
- le Silhouette Score
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

df = pd.read_csv(
    "data/intermediate/scaled_features.csv"
)

print("=" * 60)
print("DONNÉES")
print("=" * 60)

print("Dimensions originales :", df.shape)
print("\nColonnes disponibles :")
print(list(df.columns))


# ==========================================================
# VARIABLES UTILISÉES POUR K-MEANS
# ==========================================================

features = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]

df = df[features]

print("\nVariables utilisées pour K-Means :")
print(features)

print("\nDimensions après sélection :")
print(df.shape)

print("\nAperçu des données :")
print(df.head())


# ==========================================================
# VÉRIFICATION DES VALEURS MANQUANTES
# ==========================================================

print("\n" + "=" * 60)
print("VÉRIFICATION DES DONNÉES")
print("=" * 60)

print("\nValeurs manquantes :")
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Des valeurs manquantes sont présentes dans les données."
    )

print("\nAucune valeur manquante détectée.")


# ==========================================================
# TEST DE PLUSIEURS VALEURS DE K
# ==========================================================

k_values = range(2, 11)

inertias = []
silhouettes = []

print("\n" + "=" * 60)
print("CALCUL DES INDICATEURS")
print("=" * 60)

print()

for k in k_values:

    print(f"Calcul pour K = {k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(df)

    # ------------------------------------------------------
    # Inertie
    # ------------------------------------------------------

    inertia = model.inertia_

    inertias.append(inertia)

    # ------------------------------------------------------
    # Silhouette Score
    # ------------------------------------------------------

    score = silhouette_score(
        df,
        labels
    )

    silhouettes.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f} | "
        f"Silhouette = {score:.4f}"
    )


# ==========================================================
# RÉSULTATS
# ==========================================================

print("\n" + "=" * 60)
print("RÉSULTATS")
print("=" * 60)

print("\nRésumé :")

for i, k in enumerate(k_values):

    print(
        f"K = {k} | "
        f"Inertia = {inertias[i]:.2f} | "
        f"Silhouette = {silhouettes[i]:.4f}"
    )


# ==========================================================
# MEILLEUR K SELON LE SILHOUETTE SCORE
# ==========================================================

best_k_silhouette = list(k_values)[
    silhouettes.index(max(silhouettes))
]

best_score = max(silhouettes)

print("\n" + "=" * 60)
print("MEILLEUR K SELON LE SILHOUETTE SCORE")
print("=" * 60)

print(
    f"K = {best_k_silhouette}"
)

print(
    f"Silhouette Score = {best_score:.4f}"
)


# ==========================================================
# GRAPHE ELBOW
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.title(
    "Méthode du coude (Elbow Method)"
)

plt.xlabel(
    "Nombre de clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "data/intermediate/elbow_method.png",
    dpi=300
)

plt.show()


# ==========================================================
# GRAPHE SILHOUETTE
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    list(k_values),
    silhouettes,
    marker="o"
)

plt.title(
    "Silhouette Score"
)

plt.xlabel(
    "Nombre de clusters (K)"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "data/intermediate/silhouette_score.png",
    dpi=300
)

plt.show()


# ==========================================================
# FIN
# ==========================================================

print("\n" + "=" * 60)
print("ANALYSE TERMINÉE")
print("=" * 60)

print(
    "\nGraphique Elbow enregistré : "
    "data/intermediate/elbow_method.png"
)

print(
    "Graphique Silhouette enregistré : "
    "data/intermediate/silhouette_score.png"
)

print(
    f"\nMeilleur K selon le Silhouette Score : "
    f"{best_k_silhouette}"
)

print("\nFin du programme.")