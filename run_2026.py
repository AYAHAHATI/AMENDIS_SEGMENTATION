"""
run_2026.py

Exécute le traitement 2026 SANS Airflow, avec exactement les mêmes
étapes et le même ordre que le DAG dags/amendis_pipeline.py.

Utile pour tester ou produire les résultats quand Docker/Airflow
n'est pas disponible. Pré-requis : `python main.py` (modèles entraînés).

Usage :
    python run_2026.py
"""

import time

from config.config import APPLICATION_YEAR, FILE_2026, SEGMENT_WINDOW, window_dates
from pipeline.extract import DataExtractor
from pipeline.transform import DataTransformer
from pipeline.aggregate import DataAggregator
from pipeline.feature_engineering import FeatureEngineer
from pipeline.scaler import DataScaler
from ml.predict_kmeans import predict_kmeans_2026
from ml.create_final_dataset import create_final_dataset
from ml.predict_segments import predict_segments_2026
from ml.predict_anomalies_2026 import predict_anomalies_2026


def extract_2026():
    extractor = DataExtractor()
    extractor.save_dataframe(extractor.read_txt_2026(FILE_2026), "hist_2026_extracted.csv")


def transform_2026():
    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    DataTransformer().run(
        "hist_2026_extracted.csv", "hist_2026_clean.csv",
        start_date=start, end_date=end,
    )


STEPS = [
    ("extract_2026", extract_2026),
    ("transform_2026", transform_2026),
    ("aggregate_2026", lambda: DataAggregator().run_consumption(
        "hist_2026_clean.csv", "hist_2026_client.csv")),
    ("feature_engineering_2026", lambda: FeatureEngineer().run(
        "hist_2026_client.csv", "features_2026.csv")),
    ("scaler_2026", lambda: DataScaler().run_test(
        "features_2026.csv", "scaled_features_2026.csv")),
    ("predict_2026", predict_kmeans_2026),
    ("create_final_2026", create_final_dataset),
    ("predict_segments_2026", predict_segments_2026),
    ("detect_anomalies_2026", predict_anomalies_2026),
]


def main():
    for number, (name, step) in enumerate(STEPS, start=1):
        print("\n" + "=" * 60)
        print(f"[{number}/{len(STEPS)}] {name}")
        print("=" * 60)
        started = time.time()
        step()
        print(f"-> {name} terminé en {time.time() - started:.1f} s")

    print("\nTRAITEMENT 2026 TERMINÉ : résultats dans data/final/")


if __name__ == "__main__":
    main()
