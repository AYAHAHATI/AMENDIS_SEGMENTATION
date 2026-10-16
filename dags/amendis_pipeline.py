"""
amendis_pipeline.py

Orchestration Airflow du traitement des données 2026, pour LES DEUX
RÉSEAUX (électricité et eau) en une seule exécution.

Pré-requis : les modèles ont été entraînés sur l'historique 2022-2025
avec `python main.py` (pour chaque réseau), et l'entrepôt PostgreSQL
tourne (service amendis-dwh de docker-compose.yaml).

                                       ┌── électricité (TaskGroup) ──┐
extract_2026 → transform_2026 → init_dwh                              ├→ combine_reseaux
                                       └── eau (TaskGroup) ──────────┘

- Étapes communes (une seule fois) :
    extract_2026    fichier 2026 -> couche BRONZE (Parquet)
    transform_2026  BRONZE -> SILVER (les deux réseaux, partitionnés)
    init_dwh        schéma de l'entrepôt
- Un TaskGroup par réseau, exécutés EN PARALLÈLE :
    quality_silver -> aggregate -> feature_engineering -> quality_profiles
    -> scaler -> predict (K-Means) -> create_final -> quality_results
    -> [predict_segments, detect_anomalies] -> load_dwh
- Étape finale : combine_reseaux (fichiers CSV de l'ancien Power BI)

Chaque tâche est un processus Python séparé (BashOperator) qui exécute
une étape de pipeline/steps.py avec SA variable AMENDIS_RESEAU : les deux
réseaux ne partagent aucun état en mémoire.

Un contrôle qualité BLOQUANT en échec fait échouer la tâche : les tâches
suivantes du réseau ne sont pas exécutées.

Suivi : chaque tâche enregistre une ligne dans la table suivi_executions
de l'entrepôt (durée, statut, volume, tentative) ; les contrôles qualité
vont dans controles_qualite (pipeline/monitoring.py).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

# Racine du projet : /opt/airflow dans les conteneurs (dags/ y est monté)
PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# ----------------------------------------------------------
# Résilience
# - retries : une tâche en échec à cause d'une panne PASSAGÈRE (worker pas
#   prêt, base momentanément indisponible...) est relancée automatiquement ;
# - execution_timeout : une tâche bloquée est arrêtée au bout de 30 min
#   (puis relancée) au lieu de paralyser le pipeline ;
# - les contrôles qualité ne sont PAS relancés : une erreur de données est
#   déterministe, la relancer ne ferait que retarder l'alerte.
# ----------------------------------------------------------
DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=30),
}
QUALITY_STEPS = {"quality_silver_2026", "quality_profiles_2026", "quality_results_2026"}

NETWORKS = {
    "electricite": "BASSE TENSION",
    "eau": "EAU",
}


def step_task(step, network=None, task_id=None):
    """
    Tâche qui exécute `python -m pipeline.steps <step>` dans le dossier du
    projet. Pour une étape d'un réseau, AMENDIS_RESEAU est fixé pour ce
    seul processus (append_env : les autres variables sont conservées).
    """
    extra = {"retries": 0} if step in QUALITY_STEPS else {}
    return BashOperator(
        task_id=task_id or step,
        bash_command=f'cd "{PROJECT_DIR}" && "{PYTHON}" -u -m pipeline.steps {step}',
        env={"AMENDIS_RESEAU": network} if network else None,
        append_env=True,
        **extra,
    )


with DAG(
    dag_id="amendis_pipeline",
    description="Segmentation, prédiction et anomalies 2026 - électricité et eau",
    start_date=datetime(2026, 8, 23),
    schedule="0 2 * * *",  # tous les jours à 02:00
    catchup=False,
    max_active_runs=1,     # pas deux exécutions en même temps
    default_args=DEFAULT_ARGS,
    dagrun_timeout=timedelta(hours=3),
    tags=["amendis", "etl", "data_lake", "data_warehouse", "machine_learning", "2026"],
) as dag:

    # ------------------------------------------------------
    # Étapes communes
    # ------------------------------------------------------
    extract = step_task("extract_2026")
    transform = step_task("transform_2026")
    init_dwh = step_task("init_dwh")

    # ------------------------------------------------------
    # Un groupe de tâches par réseau
    # ------------------------------------------------------
    groups = []
    for slug, network in NETWORKS.items():
        with TaskGroup(group_id=slug, tooltip=f"Traitement 2026 - {network}") as group:
            tasks = {
                step: step_task(step, network)
                for step in [
                    "quality_silver_2026",
                    "aggregate_2026",
                    "feature_engineering_2026",
                    "quality_profiles_2026",
                    "scaler_2026",
                    "predict_2026",
                    "create_final_2026",
                    "quality_results_2026",
                    "predict_segments_2026",
                    "detect_anomalies_2026",
                    "load_dwh_2026",
                ]
            }
            (
                tasks["quality_silver_2026"]
                >> tasks["aggregate_2026"]
                >> tasks["feature_engineering_2026"]
                >> tasks["quality_profiles_2026"]
                >> tasks["scaler_2026"]
                >> tasks["predict_2026"]
                >> tasks["create_final_2026"]
                >> tasks["quality_results_2026"]
                >> [tasks["predict_segments_2026"], tasks["detect_anomalies_2026"]]
                >> tasks["load_dwh_2026"]
            )
        groups.append(group)

    # ------------------------------------------------------
    # Étape finale : fichiers pour l'ancien tableau de bord Power BI
    # ------------------------------------------------------
    combine = step_task("combine_reseaux")

    extract >> transform >> init_dwh >> groups >> combine
