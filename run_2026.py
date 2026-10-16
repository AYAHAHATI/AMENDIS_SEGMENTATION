"""
run_2026.py

Exécute le traitement 2026 d'UN réseau SANS Airflow, avec les mêmes
étapes que le DAG dags/amendis_pipeline.py (pipeline/steps.py).

Réseau : variable AMENDIS_RESEAU (électricité par défaut).
Pré-requis : `python main.py` (modèles entraînés) et l'entrepôt lancé
(`docker compose up -d amendis-dwh`).

Étapes :
- communes   : extract_2026 (bronze), transform_2026 (silver), init_dwh
- du réseau  : contrôles qualité, profils, modèles, chargement entrepôt
Après les deux réseaux : `python combine_reseaux.py` (fichiers Power BI).

Le DAG Airflow, lui, traite les deux réseaux en une seule exécution.

Usage :
    python run_2026.py
"""

import sys

from pipeline.steps import SHARED_STEPS, NETWORK_STEPS, run_step


def main():
    steps = SHARED_STEPS + NETWORK_STEPS
    for number, step in enumerate(steps, start=1):
        print(f"\n[{number}/{len(steps)}]")
        if run_step(step.__name__) != 0:
            return 1
    print("\nTRAITEMENT 2026 TERMINÉ : résultats dans data/final/, data/lake/gold/ et l'entrepôt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
