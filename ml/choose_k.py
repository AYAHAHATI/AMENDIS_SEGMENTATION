"""
choose_k.py

Choix du meilleur nombre de clusters (K)
avec la méthode du coude (Elbow)
et le Silhouette Score.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================================
# Chargement des données
# ==========================================================

df = pd.read_csv("data/intermediate/scaled_features.csv")

print("=" * 60)
print("DONNÉES")
print("=" * 60)

print(df.shape)
print(df.head())


# ==========================================================
# Test de plusieurs valeurs de K
# ==========================================================

k_values = range(2, 11)

inertias = []
silhouettes = []

print("\nCalcul des indicateurs...\n")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(df)

    inertias.append(model.inertia_)

    score = silhouette_score(df, labels)

    silhouettes.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette = {score:.4f}"
    )


# ==========================================================
# Graphe Elbow
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.title("Méthode du coude (Elbow Method)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)

plt.show()


# ==========================================================
# Graphe Silhouette
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    silhouettes,
    marker="o"
)

plt.title("Silhouette Score")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Score")

plt.grid(True)

plt.show()