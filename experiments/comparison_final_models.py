"""
comparison_final_models.py

Comparaison finale des trois modèles de détection d'anomalies :

1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM

Les modèles sont comparés selon :
- nombre d'anomalies
- pourcentage d'anomalies
- temps d'exécution
- stabilité
"""

import pandas as pd


# ==========================================================
# 1. RÉSULTATS DE LA COMPARAISON INITIALE
# ==========================================================

comparison_models = pd.DataFrame({

    "MODELE": [
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ],

    "NB_ANOMALIES": [
        371,
        371,
        385
    ],

    "POURCENTAGE_ANOMALIES": [
        5.01,
        5.01,
        5.19
    ],

    "TEMPS_EXECUTION_SEC": [
        1.1073,
        0.3987,
        0.6674
    ]
})


# ==========================================================
# 2. RÉSULTATS DE ROBUSTESSE
# ==========================================================

robustness = pd.DataFrame({

    "MODELE": [
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ],

    "STABILITE_MIN": [
        89.22,
        67.12,
        57.40
    ],

    "STABILITE_MAX": [
        100.00,
        100.00,
        100.00
    ]
})


# ==========================================================
# 3. FUSION DES RÉSULTATS
# ==========================================================

final_comparison = comparison_models.merge(
    robustness,
    on="MODELE"
)


# ==========================================================
# 4. AFFICHAGE
# ==========================================================

print("=" * 90)
print("COMPARAISON FINALE DES MODÈLES DE DÉTECTION D'ANOMALIES")
print("=" * 90)

print()

print(
    final_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ==========================================================
# 5. ANALYSE
# ==========================================================

print("\n" + "=" * 90)
print("ANALYSE DES RÉSULTATS")
print("=" * 90)

print("""
Isolation Forest :
- 371 anomalies
- 5.01 %
- très bonne stabilité
- stabilité minimale : 89.22 %
- stabilité maximale : 100 %

LOF :
- 371 anomalies
- 5.01 %
- temps d'exécution le plus faible
- stabilité plus variable
- stabilité minimale : 67.12 %
- présence d'un warning concernant les valeurs dupliquées

One-Class SVM :
- 385 anomalies
- 5.19 %
- stabilité variable selon le paramètre nu
- stabilité minimale : 57.40 %
- nombre d'anomalies fortement dépendant de nu
""")


# ==========================================================
# 6. CHOIX FINAL
# ==========================================================

chosen_model = "Isolation Forest"

print("=" * 90)
print("MODÈLE RETENU")
print("=" * 90)

print(
    f"Modèle retenu : {chosen_model}"
)

print("""
Justification :

Isolation Forest présente le meilleur compromis entre
stabilité, contrôle du taux d'anomalies et robustesse
aux variations des paramètres.

Il est donc retenu pour la détection des anomalies
dans le projet Amendis.
""")


# ==========================================================
# 7. SAUVEGARDE
# ==========================================================

OUTPUT_FILE = (
    "/opt/airflow/data/final/"
    "final_anomaly_models_comparison.csv"
)

final_comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Fichier généré :")
print(OUTPUT_FILE)

print("\nComparaison finale terminée.")