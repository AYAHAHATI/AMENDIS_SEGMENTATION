"""
combine_reseaux.py

Réunit les résultats 2026 de l'électricité et de l'eau (fichiers pour
Power BI). Le code est dans pipeline/combine.py, pour être aussi
utilisable par le DAG Airflow (tâche combine_reseaux).

Usage (après avoir traité les deux réseaux) :
    python combine_reseaux.py
"""

from pipeline.combine import main

if __name__ == "__main__":
    main()
