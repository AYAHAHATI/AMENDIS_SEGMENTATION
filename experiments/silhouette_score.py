"""
silhouette_score.py

Choix du meilleur K avec le Silhouette Score.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Charger les données
df = pd.read_csv("data/intermediate/scaled_features.csv")

print("=" * 60)
print("SILHOUETTE SCORE")
print("=" * 60)

best_k = None
best_score = -1

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(df)

    score = silhouette_score(df, labels)

    print(f"K = {k}  --->  Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\n" + "=" * 60)
print("MEILLEUR K")
print("=" * 60)

print(f"K optimal : {best_k}")
print(f"Silhouette : {best_score:.4f}")