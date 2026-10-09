"""
predict_segments.py

Prédiction supervisée du segment futur d'un contrat.

Principe :
- variables explicatives : profil septembre-décembre de l'année N
  (4 mois, standardisé avec le scaler historique) ;
- cible : segment K-Means du profil janvier-avril de l'année N+1.

Les deux fenêtres font 4 mois, comme les données 2026 (janvier-avril),
donc les variables restent comparables entre apprentissage et application.

Validation temporelle :
- apprentissage : 2022->2023 et 2023->2024
- test          : 2024->2025
Puis réentraînement sur les 3 transitions et application :
septembre-décembre 2025 -> janvier-avril 2026.

Deux baselines servent de référence :
- "Classe majoritaire" : prédit toujours le segment le plus fréquent ;
- "Persistance"        : prédit que le segment ne change pas
                         (segment K-Means du profil septembre-décembre).

Deux points d'entrée :
- train_segment_model()   : comparaison + réentraînement (main.py) ;
- predict_segments_2026() : application 2026 (tâche Airflow).
"""

import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config.config import (
    INTERMEDIATE_DIR,
    FINAL_DIR,
    ID_COLUMN,
    MODEL_FEATURES,
    SCALER_FILE,
    KMEANS_FILE,
    SEGMENT_MODEL_FILE,
    SEGMENT_WINDOW,
    PAST_WINDOW,
    TRAIN_TRANSITIONS,
    TEST_TRANSITIONS,
    APPLICATION_YEAR,
    RANDOM_STATE,
    window_dates,
    ensure_dirs,
)
from ml.labels import load_labels
from pipeline.history import load_history, aggregate_window

warnings.filterwarnings("ignore", category=UserWarning)

PAST_PROFILE_FILE = (
    INTERMEDIATE_DIR / f"profil_sep_dec_{APPLICATION_YEAR - 1}.csv"
)


# ==========================================================
# OUTILS
# ==========================================================

def segment_of(profile, scaler, kmeans, labels):
    """Segment K-Means (libellé) d'un profil de 4 mois."""
    clusters = kmeans.predict(scaler.transform(profile[MODEL_FEATURES]))
    return pd.Series(clusters).map(labels).values


def complete(profile):
    return profile[profile["PROFIL_COMPLET"]]


def build_transition(history, scaler, kmeans, labels, past_year, future_year):
    past = complete(aggregate_window(history, *window_dates(past_year, PAST_WINDOW)))
    future = complete(
        aggregate_window(history, *window_dates(future_year, SEGMENT_WINDOW))
    )

    merged = past.merge(future, on=ID_COLUMN, suffixes=("_PAST", "_FUTUR"))
    print(
        f"Transition {past_year}->{future_year} : passé {len(past):,} | "
        f"futur {len(future):,} | communs {len(merged):,}"
    )

    past_features = merged[[f"{f}_PAST" for f in MODEL_FEATURES]].set_axis(
        MODEL_FEATURES, axis=1
    )
    future_features = merged[[f"{f}_FUTUR" for f in MODEL_FEATURES]].set_axis(
        MODEL_FEATURES, axis=1
    )

    result = pd.DataFrame(
        scaler.transform(past_features), columns=MODEL_FEATURES
    )
    result.insert(0, ID_COLUMN, merged[ID_COLUMN].values)
    result["SEGMENT_ACTUEL"] = segment_of(past_features, scaler, kmeans, labels)
    result["SEGMENT_FUTUR"] = segment_of(future_features, scaler, kmeans, labels)
    result["TRANSITION"] = f"{past_year}->{future_year}"
    return result


def metrics(y_true, y_pred, name):
    return {
        "MODELE": name,
        "ACCURACY": accuracy_score(y_true, y_pred),
        "PRECISION_WEIGHTED": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "RECALL_WEIGHTED": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_WEIGHTED": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "PRECISION_MACRO": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "RECALL_MACRO": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_MACRO": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def baselines(y_train, y_true, persistence):
    majority = y_train.value_counts().idxmax()
    return [
        metrics(y_true, [majority] * len(y_true), "Baseline - classe majoritaire"),
        metrics(y_true, persistence, "Baseline - persistance"),
    ]


def candidate_models():
    # class_weight="balanced" : les segments rares pèsent autant que
    # le segment majoritaire. Cela augmente le rappel des petites
    # classes mais fait sur-prédire ces classes (précision plus faible).
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        # Même modèle sans pondération : à comparer, car la pondération
        # peut faire sur-prédire les petits segments.
        "Logistic Regression (sans pondération)": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            C=1.0, max_iter=5000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# ==========================================================
# ENTRAÎNEMENT ET COMPARAISON
# ==========================================================

def train_segment_model(history=None):
    ensure_dirs()
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    labels = load_labels()

    if history is None:
        history = load_history()

    print("\nConstruction des transitions (4 mois -> 4 mois)")
    train = pd.concat(
        [build_transition(history, scaler, kmeans, labels, p, f)
         for p, f in TRAIN_TRANSITIONS],
        ignore_index=True,
    )
    test = pd.concat(
        [build_transition(history, scaler, kmeans, labels, p, f)
         for p, f in TEST_TRANSITIONS],
        ignore_index=True,
    )
    print(f"Apprentissage : {len(train):,} | test temporel : {len(test):,}")

    X_train, y_train = train[MODEL_FEATURES], train["SEGMENT_FUTUR"]
    X_test, y_test = test[MODEL_FEATURES], test["SEGMENT_FUTUR"]

    results = baselines(y_train, y_test, test["SEGMENT_ACTUEL"])
    trained = {}

    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        trained[name] = model
        results.append(metrics(y_test, predictions, name))
        print(f"\n===== {name} =====")
        print(classification_report(y_test, predictions, zero_division=0))

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        FINAL_DIR / "comparaison_modeles_segments.csv",
        index=False, encoding="utf-8-sig",
    )
    print("\nCOMPARAISON (test temporel 2024->2025)")
    print(comparison.round(4).to_string(index=False))

    ml_only = comparison[~comparison["MODELE"].str.startswith("Baseline")]
    best_name = ml_only.sort_values("F1_MACRO", ascending=False).iloc[0]["MODELE"]
    print(f"\nModèle retenu (meilleur F1 macro) : {best_name}")

    # Réentraînement sur toutes les transitions historiques
    all_history = pd.concat([train, test], ignore_index=True)
    final_model = candidate_models()[best_name]
    final_model.fit(all_history[MODEL_FEATURES], all_history["SEGMENT_FUTUR"])
    joblib.dump({"name": best_name, "model": final_model}, SEGMENT_MODEL_FILE)
    print(f"Modèle sauvegardé : {SEGMENT_MODEL_FILE}")

    # Profil septembre-décembre de l'année précédant 2026,
    # sauvegardé pour que le DAG n'ait pas à relire tout l'historique
    past = aggregate_window(
        history, *window_dates(APPLICATION_YEAR - 1, PAST_WINDOW)
    )
    past.to_csv(PAST_PROFILE_FILE, index=False)
    print(f"Profil passé enregistré : {PAST_PROFILE_FILE} ({len(past):,})")

    return comparison


# ==========================================================
# APPLICATION 2026
# ==========================================================

def predict_segments_2026(reference_file="clients_cluster_2026.csv"):
    """
    Prédit le segment janvier-avril 2026 à partir du profil
    septembre-décembre 2025, puis compare au segment K-Means 2026.
    """
    for path in (SEGMENT_MODEL_FILE, SCALER_FILE, KMEANS_FILE, PAST_PROFILE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path} (lancer main.py)")

    bundle = joblib.load(SEGMENT_MODEL_FILE)
    model, model_name = bundle["model"], bundle["name"]
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    labels = load_labels()

    past = complete(pd.read_csv(PAST_PROFILE_FILE))
    reference = pd.read_csv(INTERMEDIATE_DIR / reference_file)
    reference = reference[reference["CLUSTER"] >= 0]  # profils complets

    data = past.merge(
        reference[[ID_COLUMN, "LIBELLE_CLUSTER"]], on=ID_COLUMN
    ).rename(columns={"LIBELLE_CLUSTER": "SEGMENT_REFERENCE_2026"})
    print(
        f"Contrats Sep-Dec {APPLICATION_YEAR - 1} : {len(past):,} | "
        f"segmentés 2026 : {len(reference):,} | communs : {len(data):,}"
    )

    X = scaler.transform(data[MODEL_FEATURES])
    data["SEGMENT_PREDIT_2026"] = model.predict(X)
    data["SEGMENT_SEP_DEC_2025"] = segment_of(data, scaler, kmeans, labels)
    data["CORRECT"] = data["SEGMENT_PREDIT_2026"] == data["SEGMENT_REFERENCE_2026"]

    y_true = data["SEGMENT_REFERENCE_2026"]
    majority = y_true.value_counts().idxmax()
    results = pd.DataFrame([
        metrics(y_true, [majority] * len(y_true), "Baseline - classe majoritaire"),
        metrics(y_true, data["SEGMENT_SEP_DEC_2025"], "Baseline - persistance"),
        metrics(y_true, data["SEGMENT_PREDIT_2026"], model_name),
    ])
    results.insert(1, "CONTRATS_EVALUES", len(data))

    print(f"\nRÉSULTATS 2026 (référence = K-Means janvier-avril 2026)")
    print(results.round(4).to_string(index=False))
    print(classification_report(y_true, data["SEGMENT_PREDIT_2026"], zero_division=0))

    confusion = pd.crosstab(
        data["SEGMENT_REFERENCE_2026"], data["SEGMENT_PREDIT_2026"],
        rownames=["SEGMENT_REFERENCE_2026"], colnames=["SEGMENT_PREDIT_2026"],
    )
    print(confusion.to_string())

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    data[[ID_COLUMN, "SEGMENT_SEP_DEC_2025", "SEGMENT_PREDIT_2026",
          "SEGMENT_REFERENCE_2026", "CORRECT"]].to_csv(
        FINAL_DIR / "predictions_segments_2026.csv", index=False
    )
    results.to_csv(FINAL_DIR / "metriques_prediction_2026.csv", index=False)
    confusion.to_csv(FINAL_DIR / "matrice_confusion_2026.csv")
    return data


if __name__ == "__main__":
    train_segment_model()
