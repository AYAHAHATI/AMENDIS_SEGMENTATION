"""
choose_k.py

Choix du nombre de clusters sur les profils janvier-avril 2022-2025 :
- méthode du coude (inertie, sur tous les profils) ;
- Silhouette Score (sur un échantillon aléatoire, car son coût
  est quadratique en nombre de profils).

Sorties : data/intermediate/elbow_method.png,
          data/intermediate/silhouette_score.png,
          data/intermediate/choix_k.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config.config import INTERMEDIATE_DIR, MODEL_FEATURES, RANDOM_STATE

INPUT_FILE = INTERMEDIATE_DIR / "training_windows.csv"
SILHOUETTE_SAMPLE = 20_000


def choose_k(k_values=range(2, 11)):
    df = pd.read_csv(INPUT_FILE)
    df = df[df["PROFIL_COMPLET"]]
    X = StandardScaler().fit_transform(df[MODEL_FEATURES])
    sample_size = min(SILHOUETTE_SAMPLE, len(X))

    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(
            X, labels, sample_size=sample_size, random_state=RANDOM_STATE
        )
        rows.append({"K": k, "INERTIE": model.inertia_, "SILHOUETTE": score})
        print(f"K={k:2d} | inertie={model.inertia_:12,.1f} | silhouette={score:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(INTERMEDIATE_DIR / "choix_k.csv", index=False)

    for column, title, filename in [
        ("INERTIE", "Méthode du coude (Elbow Method)", "elbow_method.png"),
        ("SILHOUETTE", "Silhouette Score", "silhouette_score.png"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(result["K"], result[column], marker="o")
        plt.title(title)
        plt.xlabel("Nombre de clusters (K)")
        plt.ylabel(column.capitalize())
        plt.xticks(result["K"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(INTERMEDIATE_DIR / filename, dpi=300)
        plt.close()

    best = result.loc[result["SILHOUETTE"].idxmax()]
    print(f"\nMeilleur Silhouette : K={int(best.K)} ({best.SILHOUETTE:.3f})")
    return result


if __name__ == "__main__":
    choose_k()
