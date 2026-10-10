# Amendis – Segmentation de la clientèle

Projet de fin d'année (INSEA, Data and Software Engineering) : segmentation des contrats Amendis selon leur consommation, détection des comportements atypiques et prédiction du segment futur, orchestrées avec Apache Airflow et restituées dans Power BI.

## Principe

Les données 2026 couvrent du 1er janvier au 18 mai 2026. Mai est incomplet, donc **tout le projet travaille sur des fenêtres de 4 mois** : chaque profil de consommation est calculé sur janvier–avril, aussi bien pour l'entraînement (2022 à 2025) que pour l'application (2026). Les variables restent ainsi comparables entre l'apprentissage et l'application.

| Étape | Données | Détail |
|---|---|---|
| Profils d'entraînement | janvier–avril 2022, 2023, 2024, 2025 | un profil par contrat et par année |
| Segmentation | K-Means (K = 5) | sur CONSO_TOTALE, CONSO_MOYENNE, CONSO_MAX, CONSO_MIN |
| Prédiction du segment | septembre–décembre N → janvier–avril N+1 | apprentissage 2022→23 et 2023→24, test 2024→25, application 2025→26 |
| Anomalies | Isolation Forest (modèle global) | contamination 5 %, 200 arbres |

Règles de préparation, identiques partout :
- un modèle par réseau : électricité (`BASSE TENSION`, kWh) et eau (m³) ne sont jamais mélangées ; chaque réseau a ses dossiers `data/intermediate/<reseau>`, `data/final/<reseau>` et `models/<reseau>` (`electricite` ou `eau`) ;
- les relevés sans volume ou sans date sont supprimés (aucune imputation) ;
- les lignes d'un même contrat à la même date sont additionnées, car plusieurs registres apparaissent souvent, dont une ligne à 0 ;
- un profil avec moins de 3 relevés sur la fenêtre est « Profil incomplet » et n'est pas segmenté ;
- NB_RELEVES est calculé et conservé pour le reporting, mais il n'entre pas dans les modèles : sur 4 mois, il vaut presque toujours 4.

## Structure

```
config/config.py        paramètres (chemins, fenêtres, réseau, K, modèles)
pipeline/               extraction, nettoyage, agrégation, variables, standardisation
ml/                     entraînement et application des modèles
dags/amendis_pipeline.py  DAG Airflow (traitement 2026)
experiments/            scripts exploratoires de la version précédente (non maintenus)
data/raw/               données Amendis complètes (non versionnées)
data/sample/            échantillons pour tester rapidement
models/                 modèles entraînés (non versionnés)
```

## Utilisation

### 1. Entraînement sur l'historique 2022–2025

```bash
pip install -r requirements.txt
python main.py --choose-k        # données complètes dans data/raw + étude du nombre de clusters
python main.py --sample          # test rapide sur data/sample
```

Pour le réseau eau (Windows, cmd) :

```bat
set AMENDIS_RESEAU=EAU
python -u main.py
python -u run_2026.py
set AMENDIS_RESEAU=
```

Puis, une fois les deux réseaux traités :

```bash
python combine_reseaux.py   # data/final/tous_reseaux/ + fichiers au format de l'ancien Power BI dans data/final/
```

Fichiers attendus dans `data/raw/` : `HIST_CSO_STG22_24.csv`, `HIST_CSO.csv`, `HIST_CSO_STG2026.txt`.

Sorties :
- `models/` : scaler, K-Means, libellés des segments, modèle de prédiction, Isolation Forest ;
- `data/final/comparaison_modeles_segments.csv` : comparaison des modèles, avec les deux baselines.

### 2. Traitement 2026 avec Airflow

Docker Desktop doit être lancé (« Engine running »). Le fichier `.env` n'est à créer qu'une fois.

```bash
# Windows (cmd)
echo AIRFLOW_UID=50000> .env
# Linux / macOS
echo "AIRFLOW_UID=$(id -u)" > .env

docker compose up -d --build
```

Ouvrir http://localhost:8080 (airflow / airflow), puis lancer le DAG `amendis_pipeline` :

```
extract → transform (janvier–avril 2026) → aggregate → feature_engineering → scaler
→ predict (K-Means) → create_final → predict_segments_2026 + detect_anomalies_2026
```

Sans Docker, les mêmes 9 étapes s'exécutent avec Python seul :

```bash
python run_2026.py
```

### 3. Fichiers pour Power BI (`data/final/<reseau>/` et `data/final/tous_reseaux/`)

Dans les fichiers combinés, la colonne `RESEAU` indique le réseau : filtrer par réseau, sans additionner kWh et m³.

| Fichier | Contenu |
|---|---|
| `clients_segmentes_2026.csv` | profil janvier–avril 2026 et segment de chaque contrat |
| `predictions_segments_2026.csv` | segment prédit, segment de référence, baseline de persistance |
| `metriques_prediction_2026.csv` | métriques 2026 du modèle et des baselines |
| `matrice_confusion_2026.csv` | matrice de confusion |
| `anomaly_scores_2026.csv` | indicateur et score d'anomalie (vide = non évalué) |

## Lecture des résultats

- Le segment « de référence » 2026 est celui que le K-Means attribue au profil janvier–avril 2026. Ce n'est pas une vérité terrain externe.
- La prédiction est comparée à deux baselines : la classe majoritaire et la persistance (« le segment ne change pas »). Un modèle n'apporte de la valeur que s'il fait mieux que la persistance.
- Une anomalie est un profil inhabituel par rapport à l'ensemble des contrats. Ce n'est ni une fraude ni une erreur confirmée. Le modèle global signale surtout les très gros consommateurs : c'est une limite connue.

## Confidentialité

`data/sample/` contient des extraits de données réelles, avec des identifiants hachés. Vérifier avec Amendis que leur publication est autorisée, sinon les retirer du dépôt ou le rendre privé.
