"""
steps.py

Étapes du traitement 2026, exécutables UNE PAR UNE en ligne de commande.
C'est ce module que le DAG Airflow appelle pour chaque tâche (une tâche
= un processus Python séparé, avec son propre réseau AMENDIS_RESEAU).

Trois familles d'étapes :
- communes (une fois, pour les deux réseaux) :
    extract_2026    fichier 2026 -> BRONZE
    transform_2026  BRONZE -> SILVER (électricité et eau)
    init_dwh        création du schéma de l'entrepôt
- par réseau (réseau lu dans AMENDIS_RESEAU) :
    quality_silver_2026 ... load_dwh_2026
- finale (après les deux réseaux) :
    combine_reseaux fichiers CSV au format du tableau de bord Power BI

Usage :
    python -m pipeline.steps extract_2026
    set AMENDIS_RESEAU=EAU  &&  python -m pipeline.steps aggregate_2026
    python -m pipeline.steps --list
Code retour : 0 = succès ; 1 = échec (Airflow marque alors la tâche en échec).
"""

import sys
import traceback

from config.config import APPLICATION_YEAR, FILE_2026, SEGMENT_WINDOW, NETWORK, window_dates


# ==========================================================
# DÉFINITION DES ÉTAPES
# (imports dans les fonctions : chaque étape ne charge que ce dont elle a besoin)
# ==========================================================

def extract_2026():
    from pipeline.lake import ingest_bronze
    return ingest_bronze(FILE_2026)["lignes"]


def transform_2026():
    from pipeline.lake import build_silver_2026
    return int(build_silver_2026()["RELEVES"].sum())


def init_dwh():
    from pipeline.warehouse import init_schema
    init_schema()


def quality_silver_2026():
    from pipeline.quality import quality_silver_2026 as run
    return run()


def aggregate_2026():
    from pipeline.aggregate import DataAggregator
    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    return len(DataAggregator().run_from_silver(start, end, "hist_2026_client.csv"))


def feature_engineering_2026():
    from pipeline.feature_engineering import FeatureEngineer
    return len(FeatureEngineer().run("hist_2026_client.csv", "features_2026.csv"))


def quality_profiles_2026():
    from pipeline.quality import quality_profiles_2026 as run
    return run()


def scaler_2026():
    from pipeline.scaler import DataScaler
    # Scaler historique, jamais réajusté sur 2026
    return len(DataScaler().run_test("features_2026.csv", "scaled_features_2026.csv"))


def predict_2026():
    from ml.predict_kmeans import predict_kmeans_2026
    return len(predict_kmeans_2026())


def create_final_2026():
    from ml.create_final_dataset import create_final_dataset
    return len(create_final_dataset())


def quality_results_2026():
    from pipeline.quality import quality_results_2026 as run
    return run()


def predict_segments_2026():
    from ml.predict_segments import predict_segments_2026 as run
    return len(run())


def detect_anomalies_2026():
    from ml.predict_anomalies_2026 import predict_anomalies_2026
    return len(predict_anomalies_2026())


def load_dwh_2026():
    from pipeline.warehouse import load_network, reconcile
    loaded = load_network()
    reconcile()
    return loaded["profils"]


def combine_reseaux():
    from pipeline.combine import main
    return len(main()["clients_segmentes_2026.csv"])


# Étapes communes aux deux réseaux (exécutées une seule fois)
SHARED_STEPS = [extract_2026, transform_2026, init_dwh]

# Étapes d'un réseau, dans l'ordre. Les deux étapes de la même
# « vague » (predict_segments, detect_anomalies) sont indépendantes
# et s'exécutent en parallèle dans Airflow.
NETWORK_STEPS = [
    quality_silver_2026,
    aggregate_2026,
    feature_engineering_2026,
    quality_profiles_2026,
    scaler_2026,
    predict_2026,
    create_final_2026,
    quality_results_2026,
    predict_segments_2026,
    detect_anomalies_2026,
    load_dwh_2026,
]

# Étape finale, après les deux réseaux
FINAL_STEPS = [combine_reseaux]

STEPS = {f.__name__: f for f in SHARED_STEPS + NETWORK_STEPS + FINAL_STEPS}


# ==========================================================
# EXÉCUTION
# ==========================================================

def run_step(name):
    """
    Exécute une étape, enregistre son suivi (pipeline/monitoring.py) et
    renvoie 0 (succès) ou 1 (échec : la tâche Airflow échoue).
    """
    from pipeline.monitoring import now, record_step
    from pipeline.quality import DataQualityError, QualityReport

    if name not in STEPS:
        print(f"Étape inconnue : {name}. Étapes disponibles : {', '.join(STEPS)}")
        return 1

    shared = STEPS[name] in SHARED_STEPS + FINAL_STEPS
    network = None if shared else NETWORK
    print("=" * 60)
    print(f"{name} ({'commune' if shared else NETWORK})")
    print("=" * 60, flush=True)

    started = now()
    try:
        result = STEPS[name]()
    except DataQualityError as error:
        print(f"\nPIPELINE ARRÊTÉ PAR LE CONTRÔLE QUALITÉ : {error}")
        print("Les étapes suivantes ne sont pas exécutées.")
        record_step(name, network, started, now(), "arret_qualite", message=error)
        return 1
    except Exception as error:
        traceback.print_exc()
        record_step(name, network, started, now(), "echec",
                    message=f"{type(error).__name__}: {error}")
        return 1

    rows = warnings = None
    if isinstance(result, QualityReport):
        warnings = len(result.warnings)
        rows = result.volume
    elif result is not None:
        rows = result
    ended = now()
    record_step(name, network, started, ended, "succes", rows=rows, warnings=warnings)
    print(f"-> {name} terminé en {(ended - started).total_seconds():.1f} s", flush=True)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        print("\n".join(STEPS))
        return 0
    return run_step(argv[0])


if __name__ == "__main__":
    sys.exit(main())
