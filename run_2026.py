"""
run_2026.py

Exécute le traitement 2026 SANS Airflow, avec exactement les mêmes
étapes et le même ordre que le DAG dags/amendis_pipeline.py.

Utile pour tester ou produire les résultats quand Docker/Airflow
n'est pas disponible. Pré-requis : `python main.py` (modèles entraînés).

Les deux premières étapes alimentent le data lake :
- extract_2026   : fichier 2026 -> couche BRONZE (Parquet, copie fidèle)
- transform_2026 : BRONZE -> couche SILVER (relevés nettoyés,
                   partitionnés par réseau et par année)
- aggregate_2026 : SILVER (fenêtre janvier-avril) -> profils par contrat

Dernière étape : chargement de l'entrepôt PostgreSQL (schéma en étoile),
suivi d'une réconciliation gold <-> entrepôt. L'entrepôt doit tourner :
    docker compose up -d amendis-dwh

Trois contrôles qualité (pipeline/quality.py) arrêtent le traitement
si les données sont anormales : après le silver, après les profils et
après la segmentation.

Usage :
    python run_2026.py
"""

import time

from config.config import APPLICATION_YEAR, FILE_2026, SEGMENT_WINDOW, window_dates
from pipeline.lake import ingest_bronze, build_silver_2026
from pipeline.aggregate import DataAggregator
from pipeline.feature_engineering import FeatureEngineer
from pipeline.scaler import DataScaler
from ml.predict_kmeans import predict_kmeans_2026
from ml.create_final_dataset import create_final_dataset
from ml.predict_segments import predict_segments_2026
from ml.predict_anomalies_2026 import predict_anomalies_2026
from pipeline.warehouse import load_network, reconcile
from pipeline.quality import (
    DataQualityError, quality_silver_2026, quality_profiles_2026, quality_results_2026,
)


def aggregate_2026():
    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    DataAggregator().run_from_silver(start, end, "hist_2026_client.csv")


STEPS = [
    ("extract_2026", lambda: ingest_bronze(FILE_2026)),
    ("transform_2026", build_silver_2026),
    ("quality_silver_2026", quality_silver_2026),
    ("aggregate_2026", aggregate_2026),
    ("feature_engineering_2026", lambda: FeatureEngineer().run(
        "hist_2026_client.csv", "features_2026.csv")),
    ("quality_profiles_2026", quality_profiles_2026),
    ("scaler_2026", lambda: DataScaler().run_test(
        "features_2026.csv", "scaled_features_2026.csv")),
    ("predict_2026", predict_kmeans_2026),
    ("create_final_2026", create_final_dataset),
    ("quality_results_2026", quality_results_2026),
    ("predict_segments_2026", predict_segments_2026),
    ("detect_anomalies_2026", predict_anomalies_2026),
    ("load_dwh_2026", lambda: (load_network(), reconcile())),
]


def main():
    for number, (name, step) in enumerate(STEPS, start=1):
        print("\n" + "=" * 60)
        print(f"[{number}/{len(STEPS)}] {name}")
        print("=" * 60)
        started = time.time()
        try:
            step()
        except DataQualityError as error:
            print(f"\nPIPELINE ARRÊTÉ PAR LE CONTRÔLE QUALITÉ : {error}")
            print("Les étapes suivantes ne sont pas exécutées.")
            raise SystemExit(1)
        print(f"-> {name} terminé en {time.time() - started:.1f} s")

    print("\nTRAITEMENT 2026 TERMINÉ : résultats dans data/final/ et data/lake/gold/")


if __name__ == "__main__":
    main()
