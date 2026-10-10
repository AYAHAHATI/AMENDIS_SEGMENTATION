"""
config.py

Paramètres centraux du projet Amendis.

Toutes les étapes (entraînement, prédiction, Airflow) lisent
leurs chemins, fenêtres temporelles et variables ici, afin que
les modèles soient toujours appliqués sur des données construites
exactement de la même façon que celles d'entraînement.
"""

import os
from pathlib import Path


# ==========================================================
# CHEMINS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = BASE_DIR / "models"


# ==========================================================
# MODE DE DONNÉES
# ----------------------------------------------------------
# "raw"    : fichiers complets dans data/raw (usage réel)
# "sample" : échantillons dans data/sample (tests rapides)
# ==========================================================

DATA_MODE = os.environ.get("AMENDIS_DATA_MODE", "raw")

if DATA_MODE == "sample":
    SOURCE_DIR = DATA_DIR / "sample"
    HIST_FILES = [
        # (fichier, colonne identifiant, date début, date fin)
        ("HIST_CSO_STG22_24_SAMPLE.csv", "NUM_CONTRAT_HASH",
         "2022-01-01", "2023-12-31"),
        ("HIST_CSO_SAMPLE.csv", "NUM_CTA_HASH",
         "2024-01-01", "2025-12-31"),
    ]
else:
    SOURCE_DIR = DATA_DIR / "raw"
    HIST_FILES = [
        ("HIST_CSO_STG22_24.csv", "NUM_CONTRAT_HASH",
         "2022-01-01", "2023-12-31"),
        ("HIST_CSO.csv", "NUM_CTA_HASH",
         "2024-01-01", "2025-12-31"),
    ]

FILE_2026 = os.environ.get(
    "AMENDIS_FILE_2026",
    "HIST_CSO_STG2026.txt"
)


# ==========================================================
# COLONNES
# ==========================================================

ID_COLUMN = "NUM_CTA_HASH"
VOLUME_COLUMN = "VOL_CONSO"
DATE_COLUMN = "DAT_PRE_RLV_CSO"
NETWORK_COLUMN = "TYPE_RESEAUX"

# Variables du profil de consommation (calculées et conservées)
FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES",
]

# Variables utilisées par les modèles.
# Sur une fenêtre de 4 mois, NB_RELEVES vaut presque toujours 4 :
# il ne décrit pas le comportement mais seulement la complétude
# du profil. Il sert donc de filtre (MIN_RELEVES), pas de variable.
MODEL_FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
]

# Un profil janvier-avril est "complet" s'il a au moins 3 relevés.
# Les profils incomplets (contrat ouvert/résilié en cours de fenêtre)
# ne sont pas segmentés : leur total serait sous-estimé.
MIN_RELEVES = 3
INCOMPLETE_CLUSTER = -1
INCOMPLETE_LABEL = "Profil incomplet"


# ==========================================================
# RÉSEAU ANALYSÉ
# ----------------------------------------------------------
# L'électricité (kWh) et l'eau (m3) ne sont pas comparables :
# un modèle est construit pour un seul réseau à la fois.
# Pour l'eau : AMENDIS_RESEAU=EAU
# ==========================================================

NETWORK = os.environ.get("AMENDIS_RESEAU", "BASSE TENSION")


# ==========================================================
# FENÊTRES TEMPORELLES (4 MOIS)
# ----------------------------------------------------------
# Les données 2026 vont du 1er janvier au 18 mai 2026.
# Mai est incomplet : on travaille sur janvier-avril.
# Toutes les variables (entraînement et application) sont donc
# calculées sur des fenêtres de 4 mois, pour rester comparables.
# ==========================================================

# Fenêtre de segmentation (janvier-avril)
SEGMENT_WINDOW = ("01-01", "04-30")

# Années historiques utilisées pour entraîner le K-Means
TRAINING_YEARS = [2022, 2023, 2024, 2025]

# Année d'application
APPLICATION_YEAR = 2026

# Fenêtre "passée" utilisée pour prédire le segment suivant
# (septembre-décembre de l'année précédente, 4 mois également)
PAST_WINDOW = ("09-01", "12-31")

# Transitions pour la prédiction supervisée :
# (année du passé, année du futur)
TRAIN_TRANSITIONS = [(2022, 2023), (2023, 2024)]
TEST_TRANSITIONS = [(2024, 2025)]


# ==========================================================
# PARAMÈTRES DES MODÈLES
# ==========================================================

N_CLUSTERS = int(os.environ.get("AMENDIS_K", 5))
RANDOM_STATE = 42

ISOLATION_FOREST_TREES = 200
ISOLATION_FOREST_CONTAMINATION = 0.05



# ==========================================================
# FICHIERS DE MODÈLES
# ==========================================================

SCALER_FILE = MODELS_DIR / "scaler.pkl"
KMEANS_FILE = MODELS_DIR / f"kmeans_model_k{N_CLUSTERS}.pkl"
LABELS_FILE = MODELS_DIR / "cluster_labels.json"
SEGMENT_MODEL_FILE = MODELS_DIR / "segment_prediction_model.pkl"
ANOMALY_MODEL_FILE = MODELS_DIR / "isolation_forest_models.joblib"


def window_dates(year, window):
    """Retourne (date_début, date_fin) d'une fenêtre pour une année."""
    return f"{year}-{window[0]}", f"{year}-{window[1]}"


def ensure_dirs():
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for path in (INTERMEDIATE_DIR, FINAL_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
