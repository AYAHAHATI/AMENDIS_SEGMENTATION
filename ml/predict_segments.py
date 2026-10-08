# ============================================================
# AMENDIS - PREDICTION DES SEGMENTS 2026
# VERSION AVEC F1 MACRO + LINEAR SVM
# ============================================================

from pathlib import Path
import warnings
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HIST_2022_2023 = (
    ROOT / "data" / "raw" / "HIST_CSO_STG22_24.csv"
)

HIST_2024_2025 = (
    ROOT / "data" / "raw" / "HIST_CSO.csv"
)

HIST_2026 = (
    ROOT / "data" / "intermediate" / "hist_2026_client.csv"
)

SCALER_FILE = (
    ROOT / "models" / "scaler.pkl"
)

KMEANS_FILE = (
    ROOT / "models" / "kmeans_model_k5.pkl"
)

FINAL_DIR = (
    ROOT / "data" / "final"
)

MODEL_DIR = (
    ROOT / "models"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "CONSO_TOTALE",
    "CONSO_MOYENNE",
    "CONSO_MAX",
    "CONSO_MIN",
    "NB_RELEVES",
]


# ============================================================
# 3. LABELS K-MEANS
# ============================================================

LABEL_MAP = {
    0: "Faible consommation",
    1: "Consommation moyenne",
    4: "Forte consommation",
    3: "Très forte consommation",
    2: "Cluster 2",
}


# ============================================================
# 4. CHARGEMENT DES DONNEES HISTORIQUES
# ============================================================

def load_historical_data():

    print("\n" + "=" * 70)
    print("CHARGEMENT DES DONNEES HISTORIQUES")
    print("=" * 70)

    # ========================================================
    # 2022-2023
    # ========================================================

    print(
        f"\nLecture : {HIST_2022_2023.name}"
    )

    if not HIST_2022_2023.exists():

        raise FileNotFoundError(
            f"Fichier introuvable :\n"
            f"{HIST_2022_2023}"
        )

    pieces_2022_2023 = []

    for chunk in pd.read_csv(
        HIST_2022_2023,
        sep="\t",
        encoding="cp1252",
        usecols=[
            "NUM_CONTRAT_HASH",
            "VOL_CONSO",
            "DAT_PRE_RLV_CSO",
        ],
        chunksize=200000,
        low_memory=False,
    ):

        chunk["DAT_PRE_RLV_CSO"] = pd.to_datetime(
            chunk["DAT_PRE_RLV_CSO"],
            dayfirst=True,
            errors="coerce",
        )

        chunk["VOL_CONSO"] = pd.to_numeric(
            chunk["VOL_CONSO"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=[
                "NUM_CONTRAT_HASH",
                "DAT_PRE_RLV_CSO",
                "VOL_CONSO",
            ]
        )

        chunk = chunk[
            (chunk["DAT_PRE_RLV_CSO"] >= pd.Timestamp("2022-01-01"))
            &
            (chunk["DAT_PRE_RLV_CSO"] <= pd.Timestamp("2023-12-31"))
        ]

        if len(chunk) > 0:
            pieces_2022_2023.append(chunk)

    if not pieces_2022_2023:

        raise ValueError(
            "Aucune donnée 2022-2023 trouvée."
        )

    df_2022_2023 = pd.concat(
        pieces_2022_2023,
        ignore_index=True
    )

    df_2022_2023 = df_2022_2023.rename(
        columns={
            "NUM_CONTRAT_HASH":
                "NUM_CTA_HASH"
        }
    )

    print(
        "Lignes 2022-2023 conservées : "
        f"{len(df_2022_2023):,}"
    )

    # ========================================================
    # 2024-2025
    # ========================================================

    print(
        f"\nLecture : {HIST_2024_2025.name}"
    )

    if not HIST_2024_2025.exists():

        raise FileNotFoundError(
            f"Fichier introuvable :\n"
            f"{HIST_2024_2025}"
        )

    pieces_2024_2025 = []

    for chunk in pd.read_csv(
        HIST_2024_2025,
        sep="\t",
        encoding="cp1252",
        usecols=[
            "NUM_CTA_HASH",
            "VOL_CONSO",
            "DAT_PRE_RLV_CSO",
        ],
        chunksize=200000,
        low_memory=False,
    ):

        chunk["DAT_PRE_RLV_CSO"] = pd.to_datetime(
            chunk["DAT_PRE_RLV_CSO"],
            dayfirst=True,
            errors="coerce",
        )

        chunk["VOL_CONSO"] = pd.to_numeric(
            chunk["VOL_CONSO"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=[
                "NUM_CTA_HASH",
                "DAT_PRE_RLV_CSO",
                "VOL_CONSO",
            ]
        )

        chunk = chunk[
            (chunk["DAT_PRE_RLV_CSO"] >= pd.Timestamp("2024-01-01"))
            &
            (chunk["DAT_PRE_RLV_CSO"] <= pd.Timestamp("2025-12-31"))
        ]

        if len(chunk) > 0:
            pieces_2024_2025.append(chunk)

    if not pieces_2024_2025:

        raise ValueError(
            "Aucune donnée 2024-2025 trouvée."
        )

    df_2024_2025 = pd.concat(
        pieces_2024_2025,
        ignore_index=True
    )

    print(
        "Lignes 2024-2025 conservées : "
        f"{len(df_2024_2025):,}"
    )

    # ========================================================
    # FUSION 2022-2025
    # ========================================================

    df = pd.concat(
        [
            df_2022_2023,
            df_2024_2025
        ],
        ignore_index=True
    )

    print(
        f"\nTotal lignes historiques : "
        f"{len(df):,}"
    )

    print(
        "Période disponible : "
        f"{df['DAT_PRE_RLV_CSO'].min().date()} "
        f"-> "
        f"{df['DAT_PRE_RLV_CSO'].max().date()}"
    )

    print(
        "Clients/comptes uniques : "
        f"{df['NUM_CTA_HASH'].nunique():,}"
    )

    return df


# ============================================================
# 5. AGREGATION D'UNE PERIODE
# ============================================================

def aggregate_period(
    df,
    start_date,
    end_date
):

    part = df[
        (df["DAT_PRE_RLV_CSO"] >= pd.Timestamp(start_date))
        &
        (df["DAT_PRE_RLV_CSO"] <= pd.Timestamp(end_date))
    ].copy()

    result = (
        part
        .groupby(
            "NUM_CTA_HASH",
            as_index=False
        )
        .agg(
            CONSO_TOTALE=(
                "VOL_CONSO",
                "sum"
            ),

            CONSO_MOYENNE=(
                "VOL_CONSO",
                "mean"
            ),

            CONSO_MAX=(
                "VOL_CONSO",
                "max"
            ),

            CONSO_MIN=(
                "VOL_CONSO",
                "min"
            ),

            NB_RELEVES=(
                "VOL_CONSO",
                "count"
            ),
        )
    )

    return result


# ============================================================
# 6. CONSTRUCTION D'UNE TRANSITION
# ============================================================

def build_transition(
    df,
    scaler,
    kmeans,
    past_year,
    future_year,
    future_end="04-30"
):

    past_start = (
        f"{past_year}-09-01"
    )

    past_end = (
        f"{past_year}-12-31"
    )

    future_start = (
        f"{future_year}-01-01"
    )

    future_end_date = (
        f"{future_year}-{future_end}"
    )

    print("\n" + "-" * 70)

    print(
        f"TRANSITION {past_year} -> {future_year}"
    )

    print(
        f"Passé : {past_start} -> {past_end}"
    )

    print(
        f"Futur : {future_start} -> {future_end_date}"
    )

    # ========================================================
    # PERIODE PASSEE
    # ========================================================

    past = aggregate_period(
        df,
        past_start,
        past_end
    )

    # ========================================================
    # PERIODE FUTURE
    # ========================================================

    future = aggregate_period(
        df,
        future_start,
        future_end_date
    )

    print(
        f"Clients période passée : "
        f"{len(past):,}"
    )

    print(
        f"Clients période future : "
        f"{len(future):,}"
    )

    # ========================================================
    # CLIENTS COMMUNS
    # ========================================================

    merged = past.merge(
        future,
        on="NUM_CTA_HASH",
        suffixes=(
            "_PAST",
            "_FUTUR"
        ),
        how="inner"
    )

    print(
        f"Clients communs : "
        f"{len(merged):,}"
    )

    if len(merged) == 0:

        raise ValueError(
            f"Aucun client commun "
            f"pour {past_year}->{future_year}"
        )

    # ========================================================
    # SEGMENT FUTUR AVEC K-MEANS
    # ========================================================

    X_future = merged[
        [
            f"{feature}_FUTUR"
            for feature in FEATURES
        ]
    ].copy()

    X_future.columns = FEATURES

    X_future_scaled = scaler.transform(
        X_future
    )

    future_clusters = kmeans.predict(
        X_future_scaled
    )

    future_labels = [
        LABEL_MAP.get(
            int(cluster),
            f"Cluster {int(cluster)}"
        )
        for cluster in future_clusters
    ]

    # ========================================================
    # VARIABLES DU PASSE
    # ========================================================

    X_past = merged[
        [
            f"{feature}_PAST"
            for feature in FEATURES
        ]
    ].copy()

    X_past.columns = FEATURES

    X_past_scaled = scaler.transform(
        X_past
    )

    result = pd.DataFrame(
        X_past_scaled,
        columns=FEATURES
    )

    result.insert(
        0,
        "NUM_CTA_HASH",
        merged["NUM_CTA_HASH"].values
    )

    result["CLUSTER_FUTUR"] = (
        future_clusters
    )

    result["SEGMENT_FUTUR"] = (
        future_labels
    )

    return result


# ============================================================
# 7. EVALUATION D'UN MODELE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # IMPORTANT :
    # Macro F1 donne le même poids à chaque segment.

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)

    print(
        f"RESULTATS - {model_name}"
    )

    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision weighted : "
        f"{precision_weighted:.4f}"
    )

    print(
        f"Recall weighted    : "
        f"{recall_weighted:.4f}"
    )

    print(
        f"F1 weighted        : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"Precision macro    : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall macro       : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 macro           : "
        f"{f1_macro:.4f}"
    )

    print(
        "\nClassification report :"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "MODELE":
            model_name,

        "ACCURACY":
            accuracy,

        "PRECISION_WEIGHTED":
            precision_weighted,

        "RECALL_WEIGHTED":
            recall_weighted,

        "F1_WEIGHTED":
            f1_weighted,

        "PRECISION_MACRO":
            precision_macro,

        "RECALL_MACRO":
            recall_macro,

        "F1_MACRO":
            f1_macro,
    }


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "AMENDIS - PREDICTION DES SEGMENTS 2026"
    )

    print(
        "Version : comparaison avec F1 Macro"
    )

    print("=" * 70)


    # ========================================================
    # [1/8] VERIFICATION
    # ========================================================

    print(
        "\n[1/8] Vérification des fichiers..."
    )

    files = [
        HIST_2022_2023,
        HIST_2024_2025,
        HIST_2026,
        SCALER_FILE,
        KMEANS_FILE,
    ]

    for file_path in files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Fichier introuvable :\n"
                f"{file_path}"
            )

        print(
            f"OK : {file_path}"
        )


    # ========================================================
    # [2/8] MODELES EXISTANTS
    # ========================================================

    print(
        "\n[2/8] Chargement des modèles..."
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    kmeans = joblib.load(
        KMEANS_FILE
    )

    print(
        f"Scaler : {SCALER_FILE.name}"
    )

    print(
        f"K-Means : {KMEANS_FILE.name}"
    )

    print(
        f"Nombre de clusters : "
        f"{kmeans.n_clusters}"
    )


    # ========================================================
    # [3/8] HISTORIQUE
    # ========================================================

    print(
        "\n[3/8] Chargement historique..."
    )

    df = load_historical_data()


    # ========================================================
    # [4/8] TRANSITIONS
    # ========================================================

    print(
        "\n[4/8] Construction des transitions..."
    )

    pair_22_23 = build_transition(
        df,
        scaler,
        kmeans,
        2022,
        2023
    )

    pair_23_24 = build_transition(
        df,
        scaler,
        kmeans,
        2023,
        2024
    )

    pair_24_25 = build_transition(
        df,
        scaler,
        kmeans,
        2024,
        2025
    )


    # ========================================================
    # DATASET APPRENTISSAGE
    # ========================================================

    train_df = pd.concat(
        [
            pair_22_23,
            pair_23_24
        ],
        ignore_index=True
    )


    # ========================================================
    # DATASET TEST TEMPOREL
    # ========================================================

    test_df = pair_24_25.copy()

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "SEGMENT_FUTUR"
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "SEGMENT_FUTUR"
    ]

    print(
        "\nDataset apprentissage :"
    )

    print(
        f"Observations : "
        f"{len(X_train):,}"
    )

    print(
        "\nDataset test temporel :"
    )

    print(
        f"Observations : "
        f"{len(X_test):,}"
    )


    # ========================================================
    # [5/8] MODELES
    # ========================================================

    print(
        "\n[5/8] Entraînement des modèles..."
    )

    models = {

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),
    }


    # ========================================================
    # LINEAR SVM
    # ========================================================

    try:

        from sklearn.svm import LinearSVC

        models["Linear SVM"] = LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=5000,
            random_state=42
        )

        print(
            "Linear SVM disponible."
        )

    except Exception as e:

        print(
            "Linear SVM indisponible."
        )

        print(
            f"Cause : {type(e).__name__}"
        )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    try:

        from sklearn.ensemble import (
            RandomForestClassifier
        )

        models["Random Forest"] = (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=10,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42
            )
        )

        print(
            "Random Forest disponible."
        )

    except Exception as e:

        print(
            "Random Forest indisponible."
        )

        print(
            f"Cause : {type(e).__name__}"
        )


    # ========================================================
    # ENTRAINEMENT + EVALUATION
    # ========================================================

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print(
            f"\nEntraînement : "
            f"{model_name}"
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[
            model_name
        ] = model

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )

        results.append(
            metrics
        )


    # ========================================================
    # COMPARAISON
    # ========================================================

    comparison = pd.DataFrame(
        results
    )

    # IMPORTANT :
    # Le choix est maintenant basé sur F1 Macro.

    comparison = comparison.sort_values(
        "F1_MACRO",
        ascending=False
    ).reset_index(
        drop=True
    )

    comparison_file = (
        FINAL_DIR /
        "prediction_models_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)

    print(
        "COMPARAISON DES MODELES"
    )

    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )


    # ========================================================
    # MODELE RETENU
    # ========================================================

    best_name = comparison.iloc[0][
        "MODELE"
    ]

    best_model = trained_models[
        best_name
    ]

    print("\n" + "=" * 70)

    print(
        "MODELE RETENU POUR LA PREDICTION 2026"
    )

    print("=" * 70)

    print(
        f"Modèle : {best_name}"
    )

    print(
        f"F1 Macro : "
        f"{comparison.iloc[0]['F1_MACRO']:.4f}"
    )

    print(
        f"F1 Weighted : "
        f"{comparison.iloc[0]['F1_WEIGHTED']:.4f}"
    )


    # ========================================================
    # [6/8] RE-ENTRAINEMENT FINAL
    # ========================================================

    print(
        "\n[6/8] Ré-entraînement final..."
    )

    all_history = pd.concat(
        [
            pair_22_23,
            pair_23_24,
            pair_24_25
        ],
        ignore_index=True
    )

    X_all = all_history[
        FEATURES
    ]

    y_all = all_history[
        "SEGMENT_FUTUR"
    ]

    best_model.fit(
        X_all,
        y_all
    )

    prediction_model_file = (
        MODEL_DIR /
        "segment_prediction_model.pkl"
    )

    joblib.dump(
        best_model,
        prediction_model_file
    )

    print(
        f"Modèle sauvegardé : "
        f"{prediction_model_file}"
    )


    # ========================================================
    # [7/8] PREDICTION 2026
    # ========================================================

    print(
        "\n[7/8] Prédiction des segments 2026..."
    )

    hist_2026 = pd.read_csv(
        HIST_2026,
        low_memory=False
    )

    required_columns = [
        "NUM_CTA_HASH",
        *FEATURES
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in hist_2026.columns
    ]

    if missing_columns:

        raise ValueError(
            "Colonnes manquantes dans "
            "hist_2026_client.csv : "
            +
            ", ".join(missing_columns)
        )

    hist_2026[FEATURES] = (
        hist_2026[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    hist_2026 = hist_2026.dropna(
        subset=FEATURES
    ).copy()

    print(
        f"Clients 2026 disponibles : "
        f"{len(hist_2026):,}"
    )


    # ========================================================
    # SEPTEMBRE-DECEMBRE 2025
    # ========================================================

    past_2025 = aggregate_period(
        df,
        "2025-09-01",
        "2025-12-31"
    )

    print(
        f"Clients Sep-Dec 2025 : "
        f"{len(past_2025):,}"
    )


    # ========================================================
    # CLIENTS COMMUNS
    # ========================================================

    final_df = past_2025.merge(
        hist_2026,
        on="NUM_CTA_HASH",
        suffixes=(
            "_PAST",
            "_2026"
        ),
        how="inner"
    )

    print(
        f"Clients communs 2025/2026 : "
        f"{len(final_df):,}"
    )

    if len(final_df) == 0:

        raise ValueError(
            "Aucun client commun entre "
            "Sep-Dec 2025 et 2026."
        )


    # ========================================================
    # PREDICTION A PARTIR DU COMPORTEMENT 2025
    # ========================================================

    X_past_2025 = final_df[
        [
            f"{feature}_PAST"
            for feature in FEATURES
        ]
    ].copy()

    X_past_2025.columns = FEATURES

    X_past_2025_scaled = scaler.transform(
        X_past_2025
    )

    predicted_segments = (
        best_model.predict(
            X_past_2025_scaled
        )
    )


    # ========================================================
    # SEGMENT REEL 2026 AVEC K-MEANS
    # ========================================================

    X_actual_2026 = final_df[
        [
            f"{feature}_2026"
            for feature in FEATURES
        ]
    ].copy()

    X_actual_2026.columns = FEATURES

    X_actual_2026_scaled = scaler.transform(
        X_actual_2026
    )

    real_clusters = kmeans.predict(
        X_actual_2026_scaled
    )

    real_segments = [
        LABEL_MAP.get(
            int(cluster),
            f"Cluster {int(cluster)}"
        )
        for cluster in real_clusters
    ]


    # ========================================================
    # RESULTAT FINAL
    # ========================================================

    prediction_result = pd.DataFrame(
        {
            "NUM_CTA_HASH":
                final_df[
                    "NUM_CTA_HASH"
                ].values,

            "SEGMENT_PREDIT_2026":
                predicted_segments,

            "CLUSTER_REEL_2026":
                real_clusters,

            "SEGMENT_REEL_2026":
                real_segments,
        }
    )

    prediction_result["CORRECT"] = (
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ]
        ==
        prediction_result[
            "SEGMENT_REEL_2026"
        ]
    )


    # ========================================================
    # METRIQUES 2026
    # ========================================================

    accuracy_2026 = accuracy_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ]
    )

    precision_weighted_2026 = precision_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="weighted",
        zero_division=0
    )

    recall_weighted_2026 = recall_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="weighted",
        zero_division=0
    )

    f1_weighted_2026 = f1_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="weighted",
        zero_division=0
    )

    precision_macro_2026 = precision_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="macro",
        zero_division=0
    )

    recall_macro_2026 = recall_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="macro",
        zero_division=0
    )

    f1_macro_2026 = f1_score(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        average="macro",
        zero_division=0
    )


    # ========================================================
    # RESULTATS 2026
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "RESULTATS PREDICTION 2026"
    )

    print("=" * 70)

    print(
        f"Modèle utilisé : "
        f"{best_name}"
    )

    print(
        f"Clients évalués : "
        f"{len(prediction_result):,}"
    )

    print(
        f"Accuracy : "
        f"{accuracy_2026:.4f}"
    )

    print(
        f"Precision weighted : "
        f"{precision_weighted_2026:.4f}"
    )

    print(
        f"Recall weighted : "
        f"{recall_weighted_2026:.4f}"
    )

    print(
        f"F1 weighted : "
        f"{f1_weighted_2026:.4f}"
    )

    print(
        f"Precision macro : "
        f"{precision_macro_2026:.4f}"
    )

    print(
        f"Recall macro : "
        f"{recall_macro_2026:.4f}"
    )

    print(
        f"F1 macro : "
        f"{f1_macro_2026:.4f}"
    )


    # ========================================================
    # RAPPORT DE CLASSIFICATION 2026
    # ========================================================

    print(
        "\nClassification report 2026 :"
    )

    print(
        classification_report(
            prediction_result[
                "SEGMENT_REEL_2026"
            ],
            prediction_result[
                "SEGMENT_PREDIT_2026"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # REPARTITION REELLE
    # ========================================================

    print(
        "\nRépartition des segments RÉELS 2026 :"
    )

    print(
        prediction_result[
            "SEGMENT_REEL_2026"
        ].value_counts()
    )


    # ========================================================
    # REPARTITION PREDITE
    # ========================================================

    print(
        "\nRépartition des segments PRÉDITS 2026 :"
    )

    print(
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ].value_counts()
    )


    # ========================================================
    # [8/8] SAUVEGARDE
    # ========================================================

    print(
        "\n[8/8] Sauvegarde des résultats..."
    )


    # --------------------------------------------------------
    # Prediction 2026
    # --------------------------------------------------------

    prediction_file = (
        FINAL_DIR /
        "prediction_segments_2026.csv"
    )

    prediction_result.to_csv(
        prediction_file,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    labels = sorted(
        set(
            prediction_result[
                "SEGMENT_REEL_2026"
            ]
        )
        |
        set(
            prediction_result[
                "SEGMENT_PREDIT_2026"
            ]
        )
    )

    cm = confusion_matrix(
        prediction_result[
            "SEGMENT_REEL_2026"
        ],
        prediction_result[
            "SEGMENT_PREDIT_2026"
        ],
        labels=labels
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            f"REEL: {label}"
            for label in labels
        ],
        columns=[
            f"PREDIT: {label}"
            for label in labels
        ]
    )

    confusion_file = (
        FINAL_DIR /
        "prediction_segments_2026_confusion.csv"
    )

    confusion_df.to_csv(
        confusion_file,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # Metriques 2026
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "MODELE":
                    best_name,

                "CLIENTS_EVALUES":
                    len(prediction_result),

                "ACCURACY":
                    accuracy_2026,

                "PRECISION_WEIGHTED":
                    precision_weighted_2026,

                "RECALL_WEIGHTED":
                    recall_weighted_2026,

                "F1_WEIGHTED":
                    f1_weighted_2026,

                "PRECISION_MACRO":
                    precision_macro_2026,

                "RECALL_MACRO":
                    recall_macro_2026,

                "F1_MACRO":
                    f1_macro_2026,
            }
        ]
    )

    metrics_file = (
        FINAL_DIR /
        "prediction_2026_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # FIN
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FICHIERS CREES"
    )

    print("=" * 70)

    print(
        f"\nPrediction 2026 :\n"
        f"{prediction_file}"
    )

    print(
        f"\nMatrice de confusion :\n"
        f"{confusion_file}"
    )

    print(
        f"\nMetriques 2026 :\n"
        f"{metrics_file}"
    )

    print(
        f"\nComparaison des modèles :\n"
        f"{comparison_file}"
    )

    print(
        f"\nModèle de prédiction :\n"
        f"{prediction_model_file}"
    )

    print("\n" + "=" * 70)

    print(
        "TERMINE AVEC SUCCES"
    )

    print("=" * 70)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()