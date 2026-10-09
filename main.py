"""
main.py

Pipeline d'ENTRAÎNEMENT (historique 2022-2025).
À lancer une fois, avant le DAG Airflow qui traite 2026.

Étapes :
1. chargement et nettoyage de l'historique (un seul réseau) ;
2. profils janvier-avril 2022, 2023, 2024, 2025 ;
3. (option --choose-k) coude + Silhouette ;
4. scaler + K-Means ;
5. Isolation Forest par segment ;
6. comparaison des modèles de prédiction + modèle final.

Usage :
    python main.py                   # données complètes (data/raw)
    python main.py --sample          # échantillons (data/sample)
    python main.py --choose-k        # ajoute l'étude du nombre de clusters
    AMENDIS_RESEAU=EAU python main.py  # modèle pour le réseau eau
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement Amendis")
    parser.add_argument("--sample", action="store_true",
                        help="utiliser data/sample au lieu de data/raw")
    parser.add_argument("--choose-k", action="store_true",
                        help="calculer coude et Silhouette pour K=2..10")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample:
        # Doit être défini avant l'import de la configuration
        os.environ["AMENDIS_DATA_MODE"] = "sample"

    from config.config import DATA_MODE, NETWORK, N_CLUSTERS
    from pipeline.history import load_history
    from ml.build_training_windows import build_training_windows
    from ml.train_kmeans import train_kmeans
    from ml.detect_anomalies import train_anomaly_models
    from ml.predict_segments import train_segment_model

    print("=" * 60)
    print(f"ENTRAÎNEMENT AMENDIS | données : {DATA_MODE} | réseau : {NETWORK} | K = {N_CLUSTERS}")
    print("=" * 60)

    history = load_history()

    print("\n[1/4] Profils janvier-avril 2022-2025")
    build_training_windows(history)

    if args.choose_k:
        from ml.choose_k import choose_k
        print("\n[option] Choix du nombre de clusters")
        choose_k()

    print("\n[2/4] Scaler + K-Means")
    train_kmeans()

    print("\n[3/4] Isolation Forest par segment")
    train_anomaly_models()

    print("\n[4/4] Prédiction supervisée des segments")
    train_segment_model(history)

    print("\nENTRAÎNEMENT TERMINÉ : le DAG amendis_pipeline peut traiter 2026.")


if __name__ == "__main__":
    sys.exit(main())
