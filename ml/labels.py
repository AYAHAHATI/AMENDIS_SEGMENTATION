"""
labels.py

Libellés des segments.

Les numéros de clusters K-Means changent à chaque réentraînement :
les libellés ne sont donc plus écrits "en dur". Ils sont déduits
des centroïdes, en classant les clusters par consommation moyenne
croissante, puis enregistrés dans models/cluster_labels.json.
"""

import json

import pandas as pd

from config.config import (
    MODEL_FEATURES, LABELS_FILE, INCOMPLETE_CLUSTER, INCOMPLETE_LABEL,
)

NAMES_BY_K = {
    2: ["Faible consommation", "Forte consommation"],
    3: ["Faible consommation", "Consommation moyenne",
        "Forte consommation"],
    4: ["Faible consommation", "Consommation moyenne",
        "Forte consommation", "Très forte consommation"],
    5: ["Très faible consommation", "Faible consommation",
        "Consommation moyenne", "Forte consommation",
        "Très forte consommation"],
}


def build_labels(kmeans, scaler):
    """Construit {cluster: libellé} et le tableau des centroïdes."""
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=MODEL_FEATURES,
    )
    centroids.index.name = "CLUSTER"

    order = centroids["CONSO_MOYENNE"].sort_values().index.tolist()
    k = len(order)
    names = NAMES_BY_K.get(k, [f"Niveau {i + 1}" for i in range(k)])

    labels = {int(cluster): names[rank] for rank, cluster in enumerate(order)}
    centroids["LIBELLE_CLUSTER"] = centroids.index.map(labels)
    return labels, centroids


def save_labels(labels):
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def load_labels():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"{LABELS_FILE} introuvable : lancer ml/train_kmeans.py"
        )
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    labels[INCOMPLETE_CLUSTER] = INCOMPLETE_LABEL
    return labels
