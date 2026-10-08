import pandas as pd
import joblib
import os


# ============================================================
# PARAMÈTRES
# ============================================================

# Fichier contenant les profils de consommation 2026
INPUT_FILE = "data/final/clients_segmentes_2026.csv"

# Modèle Isolation Forest validé du projet
MODEL_FILE = (
    "data/final/"
    "isolation_forest_model_new.joblib"
)

# Scaler appris sur les données historiques 2022-2025
SCALER_FILE = (
    "data/final/"
    "isolation_forest_scaler_new.joblib"
)

# Fichier de sortie
OUTPUT_FILE = "data/final/anomaly_scores_2026.csv"


# Variables utilisées pour la détection
FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES"
]


# ============================================================
# DÉTECTION DES ANOMALIES 2026
# ============================================================

def detect_anomalies_2026():

    print("=" * 70)
    print("DÉTECTION DES ANOMALIES - DONNÉES 2026")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Chargement des données 2026
    # --------------------------------------------------------

    print("\n[1/5] Chargement des données 2026...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Nombre de clients : {len(df):,}")
    print(f"Nombre de variables : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Vérification des variables
    # --------------------------------------------------------

    print("\n[2/5] Vérification des variables...")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Variables manquantes : {missing_features}"
        )

    print("Variables utilisées :")

    for feature in FEATURES:
        print(f"  - {feature}")

    # Vérification des valeurs manquantes
    missing_values = df[FEATURES].isnull().sum()

    if missing_values.sum() > 0:

        print("\nValeurs manquantes détectées :")
        print(missing_values)

        raise ValueError(
            "Les variables utilisées pour la détection "
            "contiennent des valeurs manquantes."
        )

    print("\nAucune valeur manquante détectée.")

    # --------------------------------------------------------
    # 3. Chargement du scaler historique
    # --------------------------------------------------------

    print("\n[3/5] Chargement du scaler historique...")

    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(
            f"Scaler introuvable : {SCALER_FILE}"
        )

    scaler = joblib.load(SCALER_FILE)

    print(
        f"Scaler chargé : {SCALER_FILE}"
    )

    # Sélection des variables
    X = df[FEATURES].copy()

    # Standardisation avec le scaler historique
    X_scaled = scaler.transform(X)

    print("Standardisation terminée.")

    # --------------------------------------------------------
    # 4. Chargement du modèle Isolation Forest
    # --------------------------------------------------------

    print("\n[4/5] Chargement du modèle Isolation Forest...")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(
        f"Modèle chargé : {MODEL_FILE}"
    )

    print(
        f"Nombre d'arbres : {model.n_estimators}"
    )

    print(
        f"Contamination : {model.contamination}"
    )

    # --------------------------------------------------------
    # 5. Détection des anomalies
    # --------------------------------------------------------

    print("\n[5/5] Détection des anomalies...")

    # Isolation Forest :
    #  1  -> observation normale
    # -1  -> observation atypique

    predictions = model.predict(X_scaled)

    # Score d'anomalie
    anomaly_scores = model.decision_function(X_scaled)

    # Ajout des résultats
    df["ANOMALIE"] = predictions

    df["SCORE_ANOMALIE"] = anomaly_scores

    # Indicateur explicite :
    # 1 = anomalie
    # 0 = normal
    df["EST_ANOMALIE"] = (
        df["ANOMALIE"] == -1
    ).astype(int)

    # --------------------------------------------------------
    # Calcul des statistiques
    # --------------------------------------------------------

    nombre_anomalies = int(
        df["EST_ANOMALIE"].sum()
    )

    nombre_normaux = (
        len(df) - nombre_anomalies
    )

    pourcentage_anomalies = (
        nombre_anomalies / len(df) * 100
    )

    pourcentage_normaux = (
        nombre_normaux / len(df) * 100
    )

    # --------------------------------------------------------
    # Affichage des résultats
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉSULTATS DE LA DÉTECTION")
    print("=" * 70)

    print(
        f"Nombre total de clients : "
        f"{len(df):,}"
    )

    print(
        f"Clients normaux         : "
        f"{nombre_normaux:,} "
        f"({pourcentage_normaux:.2f} %)"
    )

    print(
        f"Clients atypiques       : "
        f"{nombre_anomalies:,} "
        f"({pourcentage_anomalies:.2f} %)"
    )

    # --------------------------------------------------------
    # Sauvegarde du résultat
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFichier sauvegardé :")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("DÉTECTION TERMINÉE")
    print("=" * 70)


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    detect_anomalies_2026()