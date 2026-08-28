"""
amendis_pipeline.py

Orchestration Airflow du pipeline de segmentation
des clients Amendis.

Le modèle Machine Learning est entraîné sur les
données historiques 2022-2025.

Les données 2026 sont utilisées comme données de test :
elles sont extraites, transformées, agrégées,
préparées puis classifiées avec le modèle historique.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


# ==========================================================
# ÉTAPE 1 : EXTRACTION 2026
# ==========================================================

def extract_2026_task():

    print("=" * 70)
    print("AIRFLOW - EXTRACTION 2026")
    print("=" * 70)

    from pipeline.extract import DataExtractor

    extractor = DataExtractor()

    df = extractor.read_txt_2026(
        "HIST_CSO_STG2026.txt"
    )

    extractor.save_dataframe(
        df,
        "hist_2026_extracted.csv"
    )

    print("Extraction 2026 terminée.")


# ==========================================================
# ÉTAPE 2 : TRANSFORMATION 2026
# ==========================================================

def transform_2026_task():

    print("=" * 70)
    print("AIRFLOW - TRANSFORMATION 2026")
    print("=" * 70)

    from pipeline.transform import DataTransformer

    transformer = DataTransformer()

    transformer.run(
        "hist_2026_extracted.csv",
        "hist_2026_clean.csv"
    )

    print("Transformation 2026 terminée.")


# ==========================================================
# ÉTAPE 3 : AGRÉGATION 2026
# ==========================================================

def aggregate_2026_task():

    print("=" * 70)
    print("AIRFLOW - AGRÉGATION 2026")
    print("=" * 70)

    import pandas as pd

    from pipeline.aggregate import DataAggregator

    # ------------------------------------------------------
    # Chargement du fichier transformé
    # ------------------------------------------------------

    input_path = (
        "/opt/airflow/data/intermediate/"
        "hist_2026_clean.csv"
    )

    print(
        f"Lecture du fichier : {input_path}"
    )

    if not pd.io.common.file_exists(input_path):

        raise FileNotFoundError(
            f"Fichier introuvable : {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Nombre de lignes avant agrégation : "
        f"{len(df)}"
    )

    print(
        f"Nombre de colonnes : "
        f"{len(df.columns)}"
    )

    print(
        f"Colonnes disponibles : "
        f"{list(df.columns)}"
    )

    # ------------------------------------------------------
    # Vérification de la clé client
    # ------------------------------------------------------

    if "NUM_CTA_HASH" not in df.columns:

        raise ValueError(
            "La colonne NUM_CTA_HASH est absente "
            "du fichier hist_2026_clean.csv."
        )

    # ------------------------------------------------------
    # Agrégation
    # ------------------------------------------------------

    aggregator = DataAggregator()

    df_aggregated = aggregator.aggregate_consumption(
        df
    )

    # ------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------

    aggregator.save_dataframe(
        df_aggregated,
        "hist_2026_client.csv"
    )

    print(
        f"Nombre de lignes après agrégation : "
        f"{len(df_aggregated)}"
    )

    print(
        f"Colonnes après agrégation : "
        f"{list(df_aggregated.columns)}"
    )

    print("Agrégation 2026 terminée.")


# ==========================================================
# ÉTAPE 4 : FEATURE ENGINEERING 2026
# ==========================================================

def feature_engineering_2026_task():

    print("=" * 70)
    print("AIRFLOW - FEATURE ENGINEERING 2026")
    print("=" * 70)

    from pipeline.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()

    engineer.run(
        "hist_2026_client.csv",
        "features_2026.csv"
    )

    print("Feature Engineering 2026 terminé.")


# ==========================================================
# ÉTAPE 5 : STANDARDISATION 2026
# ==========================================================

def scaler_2026_task():

    print("=" * 70)
    print("AIRFLOW - STANDARDISATION 2026")
    print("=" * 70)

    from pipeline.scaler import DataScaler

    scaler = DataScaler()

    # ------------------------------------------------------
    # IMPORTANT :
    # run_test utilise le scaler déjà entraîné
    # sur les données historiques 2022-2025.
    # ------------------------------------------------------

    scaler.run_test(
        "features_2026.csv",
        "scaled_features_2026.csv"
    )

    print("Standardisation 2026 terminée.")


# ==========================================================
# ÉTAPE 6 : PRÉDICTION K-MEANS 2026
# ==========================================================

def predict_2026_task():

    print("=" * 70)
    print("AIRFLOW - PRÉDICTION K-MEANS 2026")
    print("=" * 70)

    from ml.predict_kmeans import predict_kmeans_2026

    df = predict_kmeans_2026()

    print(
        f"Nombre de clients segmentés : {len(df)}"
    )

    print("Prédiction 2026 terminée.")


# ==========================================================
# ÉTAPE 7 : DATASET FINAL 2026
# ==========================================================

def create_final_2026_task():

    print("=" * 70)
    print("AIRFLOW - CRÉATION DU DATASET FINAL 2026")
    print("=" * 70)

    from ml.create_final_dataset import create_final_dataset

    df = create_final_dataset()

    print(
        f"Nombre de clients dans le dataset final : "
        f"{len(df)}"
    )

    print("Dataset final 2026 terminé.")


# ==========================================================
# DÉFINITION DU DAG
# ==========================================================

with DAG(

    dag_id="amendis_pipeline",

    description=(
        "Pipeline automatique de segmentation "
        "des clients Amendis sur les données 2026"
    ),

    start_date=datetime(
        2026,
        8,
        23
    ),

    # Exécution automatique tous les jours à 02:00
    schedule="0 2 * * *",

    catchup=False,

    tags=[
        "amendis",
        "etl",
        "machine_learning",
        "segmentation",
        "2026"
    ],

) as dag:

    # ======================================================
    # TASK 1 : EXTRACTION
    # ======================================================

    extract = PythonOperator(
        task_id="extract_2026",
        python_callable=extract_2026_task
    )

    # ======================================================
    # TASK 2 : TRANSFORMATION
    # ======================================================

    transform = PythonOperator(
        task_id="transform_2026",
        python_callable=transform_2026_task
    )

    # ======================================================
    # TASK 3 : AGRÉGATION
    # ======================================================

    aggregate = PythonOperator(
        task_id="aggregate_2026",
        python_callable=aggregate_2026_task
    )

    # ======================================================
    # TASK 4 : FEATURE ENGINEERING
    # ======================================================

    feature_engineering = PythonOperator(
        task_id="feature_engineering_2026",
        python_callable=feature_engineering_2026_task
    )

    # ======================================================
    # TASK 5 : STANDARDISATION
    # ======================================================

    scaler = PythonOperator(
        task_id="scaler_2026",
        python_callable=scaler_2026_task
    )

    # ======================================================
    # TASK 6 : PRÉDICTION
    # ======================================================

    prediction = PythonOperator(
        task_id="predict_2026",
        python_callable=predict_2026_task
    )

    # ======================================================
    # TASK 7 : DATASET FINAL
    # ======================================================

    final_dataset = PythonOperator(
        task_id="create_final_2026",
        python_callable=create_final_2026_task
    )

    # ======================================================
    # ORDRE DU PIPELINE
    # ======================================================

    (
        extract
        >> transform
        >> aggregate
        >> feature_engineering
        >> scaler
        >> prediction
        >> final_dataset
    )