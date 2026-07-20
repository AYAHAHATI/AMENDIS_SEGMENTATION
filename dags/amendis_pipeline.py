"""
amendis_pipeline.py

Orchestration du pipeline ETL Amendis avec Apache Airflow.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.extract import DataExtractor
from pipeline.transform import DataTransformer
from pipeline.aggregate import DataAggregator
from pipeline.merge import DataMerger
from pipeline.feature_engineering import FeatureEngineer
from pipeline.scaler import DataScaler


# ==========================================================
# TÂCHES DU PIPELINE
# ==========================================================

def extract_task():
    """
    Extraction des données.
    """

    extractor = DataExtractor(data_path="data/sample")

    # Historique complet : 2022-2024 + 2025
    extractor.run(
        [
            "HIST_CSO_STG22_24_SAMPLE.csv",
            "HIST_CSO_SAMPLE.csv"
        ],
        "hist_raw.csv"
    )

    # Facturation
    extractor.run(
        "FACT_STG_SAMPLE.csv",
        "fact_raw.csv"
    )


def transform_task():
    """
    Transformation des données.
    """

    transformer = DataTransformer()

    transformer.run(
        "hist_raw.csv",
        "hist_clean.csv"
    )

    transformer.run(
        "fact_raw.csv",
        "fact_clean.csv"
    )


def aggregate_task():
    """
    Agrégation des données.
    """

    aggregator = DataAggregator()

    aggregator.run(
        "hist_clean.csv",
        "fact_clean.csv",
        "hist_client.csv",
        "fact_client.csv"
    )


def merge_task():
    """
    Fusion des données.
    """

    merger = DataMerger()

    merger.run(
        "hist_client.csv",
        "fact_client.csv",
        "dataset_final.csv"
    )


def feature_task():
    """
    Feature Engineering.
    """

    engineer = FeatureEngineer()

    engineer.run(
        "dataset_final.csv",
        "features.csv"
    )


def scaler_task():
    """
    Standardisation des données.
    """

    scaler = DataScaler()

    scaler.run(
        "features.csv",
        "scaled_features.csv"
    )


# ==========================================================
# DÉFINITION DU DAG
# ==========================================================

with DAG(
    dag_id="amendis_pipeline",
    description="Pipeline ETL Amendis",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["amendis", "etl", "machine_learning"],
) as dag:

    extract = PythonOperator(
        task_id="extract",
        python_callable=extract_task
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=transform_task
    )

    aggregate = PythonOperator(
        task_id="aggregate",
        python_callable=aggregate_task
    )

    merge = PythonOperator(
        task_id="merge",
        python_callable=merge_task
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_task
    )

    scaler = PythonOperator(
        task_id="scaler",
        python_callable=scaler_task
    )

    extract >> transform >> aggregate >> merge >> feature_engineering >> scaler