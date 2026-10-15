"""
amendis_pipeline.py

Orchestration Airflow du traitement des données 2026.

Pré-requis : les modèles ont été entraînés sur l'historique
2022-2025 avec `python main.py` (dossier models/).

Les données 2026 vont du 1er janvier au 18 mai 2026. Mai étant
incomplet, seule la fenêtre janvier-avril est conservée, comme
pour les profils d'entraînement.

Data lake (data/lake/) :
- extract_2026   : fichier source -> BRONZE (Parquet, copie fidèle)
- transform_2026 : BRONZE -> SILVER (nettoyé, partitionné réseau/année)
- create_final, predict_segments, detect_anomalies -> GOLD

Contrôles qualité (pipeline/quality.py) : un contrôle BLOQUANT en échec
fait échouer la tâche, et Airflow n'exécute pas les tâches suivantes.

Enchaînement :
extract -> transform -> quality_silver -> aggregate -> feature_engineering
-> quality_profiles -> scaler -> predict (K-Means) -> create_final
-> quality_results -> [predict_segments, detect_anomalies]
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def extract_2026_task():
    """Fichier 2026 -> couche BRONZE (Parquet, copie fidèle)."""
    from config.config import FILE_2026
    from pipeline.lake import ingest_bronze

    ingest_bronze(FILE_2026)


def transform_2026_task():
    """BRONZE -> SILVER : relevés nettoyés, partitionnés par réseau et année."""
    from pipeline.lake import build_silver_2026

    build_silver_2026()


def aggregate_2026_task():
    """SILVER (janvier-avril 2026, réseau traité) -> profils par contrat."""
    from config.config import APPLICATION_YEAR, SEGMENT_WINDOW, window_dates
    from pipeline.aggregate import DataAggregator

    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    DataAggregator().run_from_silver(start, end, "hist_2026_client.csv")


def quality_silver_2026_task():
    """Contrôle qualité des relevés silver (bloquant si anomalie grave)."""
    from pipeline.quality import quality_silver_2026

    quality_silver_2026()


def quality_profiles_2026_task():
    """Contrôle qualité des profils par contrat."""
    from pipeline.quality import quality_profiles_2026

    quality_profiles_2026()


def quality_results_2026_task():
    """Contrôle qualité des segments obtenus."""
    from pipeline.quality import quality_results_2026

    quality_results_2026()


def feature_engineering_2026_task():
    from pipeline.feature_engineering import FeatureEngineer

    FeatureEngineer().run("hist_2026_client.csv", "features_2026.csv")


def scaler_2026_task():
    from pipeline.scaler import DataScaler

    # Scaler historique, jamais réajusté sur 2026
    DataScaler().run_test("features_2026.csv", "scaled_features_2026.csv")


def predict_2026_task():
    from ml.predict_kmeans import predict_kmeans_2026

    predict_kmeans_2026()


def create_final_2026_task():
    from ml.create_final_dataset import create_final_dataset

    create_final_dataset()


def predict_segments_2026_task():
    from ml.predict_segments import predict_segments_2026

    predict_segments_2026()


def detect_anomalies_2026_task():
    from ml.predict_anomalies_2026 import predict_anomalies_2026

    predict_anomalies_2026()


with DAG(
    dag_id="amendis_pipeline",
    description="Segmentation, prédiction et anomalies - données janvier-avril 2026",
    start_date=datetime(2026, 8, 23),
    schedule="0 2 * * *",  # tous les jours à 02:00
    catchup=False,
    tags=["amendis", "etl", "machine_learning", "segmentation", "anomalies", "2026"],
) as dag:

    extract = PythonOperator(task_id="extract_2026", python_callable=extract_2026_task)
    transform = PythonOperator(task_id="transform_2026", python_callable=transform_2026_task)
    quality_silver = PythonOperator(
        task_id="quality_silver_2026", python_callable=quality_silver_2026_task
    )
    aggregate = PythonOperator(task_id="aggregate_2026", python_callable=aggregate_2026_task)
    feature_engineering = PythonOperator(
        task_id="feature_engineering_2026", python_callable=feature_engineering_2026_task
    )
    quality_profiles = PythonOperator(
        task_id="quality_profiles_2026", python_callable=quality_profiles_2026_task
    )
    scaler = PythonOperator(task_id="scaler_2026", python_callable=scaler_2026_task)
    prediction = PythonOperator(task_id="predict_2026", python_callable=predict_2026_task)
    final_dataset = PythonOperator(
        task_id="create_final_2026", python_callable=create_final_2026_task
    )
    quality_results = PythonOperator(
        task_id="quality_results_2026", python_callable=quality_results_2026_task
    )
    segment_prediction = PythonOperator(
        task_id="predict_segments_2026", python_callable=predict_segments_2026_task
    )
    anomaly_detection = PythonOperator(
        task_id="detect_anomalies_2026", python_callable=detect_anomalies_2026_task
    )

    (
        extract
        >> transform
        >> quality_silver
        >> aggregate
        >> feature_engineering
        >> quality_profiles
        >> scaler
        >> prediction
        >> final_dataset
        >> quality_results
        >> [segment_prediction, anomaly_detection]
    )
