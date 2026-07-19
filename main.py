from pipeline.extract import DataExtractor
from pipeline.transform import DataTransformer
from pipeline.aggregate import DataAggregator
from pipeline.merge import DataMerger
from pipeline.feature_engineering import FeatureEngineer
from pipeline.scaler import DataScaler


def main():

    print("=" * 60)
    print("DEBUT DU PIPELINE AMENDIS")
    print("=" * 60)

    # ==========================================================
    # EXTRACTION
    # ==========================================================

    extractor = DataExtractor(data_path="data/sample")

    extractor.run(
        "HIST_CSO_SAMPLE.csv",
        "hist_raw.csv"
    )

    extractor.run(
        "FACT_STG_SAMPLE.csv",
        "fact_raw.csv"
    )

    # ==========================================================
    # TRANSFORMATION
    # ==========================================================

    transformer = DataTransformer()

    transformer.run(
        "hist_raw.csv",
        "hist_clean.csv"
    )

    transformer.run(
        "fact_raw.csv",
        "fact_clean.csv"
    )

    # ==========================================================
    # AGRÉGATION
    # ==========================================================

    aggregator = DataAggregator()

    aggregator.run(
        "hist_clean.csv",
        "fact_clean.csv",
        "hist_client.csv",
        "fact_client.csv"
    )

    # ==========================================================
    # FUSION
    # ==========================================================

    merger = DataMerger()

    merger.run(
        "hist_client.csv",
        "fact_client.csv",
        "dataset_final.csv"
    )

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    engineer = FeatureEngineer()

    engineer.run(
        "dataset_final.csv",
        "features.csv"
    )

    # ==========================================================
    # STANDARDISATION
    # ==========================================================

    scaler = DataScaler()

    scaler.run(
        "features.csv",
        "scaled_features.csv"
    )

    print("\n" + "=" * 60)
    print("PIPELINE ETL TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


if __name__ == "__main__":
    main()