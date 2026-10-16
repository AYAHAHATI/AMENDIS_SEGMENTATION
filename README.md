# Amendis – Segmentation de la clientèle

Projet de fin d'année (INSEA, Data and Software Engineering) : segmentation des contrats Amendis selon leur consommation, détection des comportements atypiques et prédiction du segment futur, orchestrées avec Apache Airflow et restituées dans Power BI.

## Principe

Les données 2026 couvrent du 1er janvier au 18 mai 2026. Mai est incomplet, donc **tout le projet travaille sur des fenêtres de 4 mois** : chaque profil de consommation est calculé sur janvier–avril, aussi bien pour l'entraînement (2022 à 2025) que pour l'application (2026). Les variables restent ainsi comparables entre l'apprentissage et l'application.

| Étape | Données | Détail |
|---|---|---|
| Profils d'entraînement | janvier–avril 2022, 2023, 2024, 2025 | un profil par contrat et par année |
| Segmentation | K-Means (K = 5) | sur CONSO_TOTALE, CONSO_MOYENNE, CONSO_MAX, CONSO_MIN |
| Prédiction du segment | septembre–décembre N → janvier–avril N+1 | apprentissage 2022→23 et 2023→24, test 2024→25, application 2025→26 |
| Anomalies | Isolation Forest (un modèle par segment) | contamination 5 %, 200 arbres |

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
dags/amendis_pipeline.py  DAG Airflow (traitement 2026, deux réseaux en parallèle)
pipeline/steps.py       étapes 2026 exécutables une par une (utilisées par le DAG)
experiments/            scripts exploratoires de la version précédente (non maintenus)
data/raw/               données Amendis complètes (non versionnées)
data/sample/            échantillons pour tester rapidement
models/                 modèles entraînés (non versionnés)
data/lake/              data lake Parquet bronze / silver / gold (non versionné, regénérable)
```

## Data lake (couches bronze / silver / gold)

```
data/raw/*.csv, *.txt                       fichiers sources
   │  pipeline.lake.ingest_bronze           copie fidèle, toutes colonnes (texte)
   ▼
data/lake/bronze/<fichier>/<fichier>.parquet   (+ _ingestion.json : lignes, tailles, durée)
   │  build_silver_history / build_silver_2026   nettoyage (mêmes règles qu'avant)
   ▼
data/lake/silver/releves/reseau=<réseau>/annee=<année>/part-0.parquet
   │  read_silver (réseau + fenêtre) -> profils, modèles
   ▼
data/lake/gold/<table>/reseau=<réseau>/part-0.parquet
   historique : profils_entrainement, segments_entrainement, statistiques_segments,
                anomalies_historique, comparaison_modeles, profils_sep_dec_2025, choix_k
   2026       : clients_segmentes_2026, anomalies_2026,
                predictions_segments_2026, metriques_prediction_2026
```

- Le premier `python main.py` construit le bronze et le silver de l'historique ; les exécutions suivantes lisent directement le silver (plus de relecture des gros CSV).
- Commandes : `python -m pipeline.lake history` (forcer : `history --force`), `python -m pipeline.lake 2026`, `python -m pipeline.lake status`.
- `python scripts/benchmark_formats.py` compare la taille et le temps de lecture CSV / Parquet.
- Les CSV de `data/final/` (lus par Power BI) sont toujours produits, à l'identique.
## Entrepôt de données PostgreSQL (`pipeline/warehouse.py`)

Schéma en étoile (`pipeline/sql/schema.sql`), alimenté à partir de la couche gold :

```
                    dim_periode (janvier-avril 2022 ... 2026)
                         |
 dim_contrat --- fait_profil_consommation --- dim_segment
                         |                    (Très faible ... Très forte, Profil incomplet)
                    dim_reseau (électricité kWh / eau m3)

 fait_prediction_segment : segment sep-déc 2025, prédit et de référence (2026)
 metriques_modeles       : test 2024-2025 et application 2026
 journal_chargements     : une ligne par chargement
 vues : v_profils, v_predictions_2026, v_synthese_segments
```

- Granularité de `fait_profil_consommation` : un contrat × une période janvier–avril (historique 2022–2025 et 2026).
- Chargement : staging (COPY) puis SQL vers les dimensions et les faits, en **une transaction** (une panne ne modifie rien), idempotent (relancer = même contenu), suivi d'une **réconciliation** gold ↔ entrepôt.
- Démarrer l'entrepôt seul : `docker compose up -d amendis-dwh` (PostgreSQL 13, même image qu'Airflow, `localhost:5433`, base `amendis_dwh`, utilisateur/mot de passe `amendis`).
- Commandes : `python -m pipeline.warehouse load` (réseau courant), `reconcile`, `status`.
- `run_2026.py` et le DAG se terminent par `load_dwh_2026`.
- Exemples de requêtes SQL : `pipeline/sql/exemples.sql`.
- Power BI : Obtenir des données → PostgreSQL → serveur `localhost:5433`, base `amendis_dwh`.

### Versions des bibliothèques : identiques partout

Les versions sont fixées exactement dans `requirements.txt` (`==`), et l'image Airflow les installe (`docker compose up -d --build`). C'est indispensable pour le Parquet : un fichier écrit par pyarrow 26 (sur le PC) ne peut pas être lu par pyarrow 19 (version par défaut de l'image Airflow 2.10.5) : erreur `OSError: Repetition level histogram size mismatch`. Après toute modification de `requirements.txt`, reconstruire l'image.

## Contrôles qualité (`pipeline/quality.py`)

Trois points de contrôle dans le traitement 2026 (tâches du DAG et de `run_2026.py`). L'historique est contrôlé avec les mêmes règles dans `main.py`, **année par année** (relevés 2022, 2023, 2024, 2025 puis profils d'apprentissage de chaque année, avec l'évolution par rapport à l'année précédente) ; un échec bloquant arrête l'entraînement.

| Tâche | Ce qui est vérifié |
|---|---|
| `quality_silver_2026` | relevés présents, aucune valeur manquante, dates dans la fenêtre, volumes négatifs, doublons, les 4 mois présents, évolution du nombre de contrats par rapport à janvier–avril de l'année précédente |
| `quality_profiles_2026` | un profil par contrat, pas de valeur manquante, MIN ≤ MOYENNE ≤ MAX, TOTALE = MOYENNE × NB_RELEVES, part de profils incomplets |
| `quality_results_2026` | tous les profils ont un segment, les 5 segments sont représentés, répartition proche de l'historique |

- **BLOQUANT** : la tâche échoue et Airflow n'exécute pas les tâches suivantes (aucun résultat faux n'est produit).
- **AVERTISSEMENT** : le traitement continue, l'écart est signalé.
- Seuils dans `config/config.py` (section CONTRÔLES QUALITÉ).
- Rapports : `data/lake/quality/dernier_<etape>_<reseau>.csv` et `historique_controles.csv` (une ligne par contrôle et par exécution).
- Démonstration : `python scripts/demo_quality.py` abîme volontairement les données en mémoire (avril absent, volumes négatifs, contrats disparus, doublons) et montre que les contrôles arrêtent le pipeline.

- Test sur échantillons : `python scripts/create_sample_2026.py` crée un faux fichier 2026 (relevés 2025 décalés d'un an), puis `AMENDIS_DATA_MODE=sample python run_2026.py`.

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

Ouvrir http://localhost:8080 (airflow / airflow), puis lancer le DAG `amendis_pipeline`. Une seule exécution traite **les deux réseaux** (26 tâches) :

```
                                        ┌── électricité (TaskGroup, 11 tâches) ──┐
extract_2026 → transform_2026 → init_dwh                                         ├→ combine_reseaux
                                        └── eau (TaskGroup, 11 tâches) ──────────┘

Dans chaque groupe :
quality_silver → aggregate → feature_engineering → quality_profiles → scaler → predict
→ create_final → quality_results → [predict_segments, detect_anomalies] → load_dwh
```

- `extract_2026` (bronze), `transform_2026` (silver des deux réseaux) et `init_dwh` (schéma de l'entrepôt) s'exécutent une seule fois.
- Les deux réseaux s'exécutent en parallèle ; chaque tâche est un processus séparé (`python -m pipeline.steps <étape>`) avec sa propre variable `AMENDIS_RESEAU`.
- `combine_reseaux` attend les deux réseaux puis produit les CSV de l'ancien tableau de bord.
- Un contrôle qualité bloquant en échec arrête les tâches suivantes du réseau (`upstream_failed`) et `combine_reseaux`.

Sans Docker, les mêmes étapes s'exécutent pour un réseau avec Python seul :

```bash
python run_2026.py                       # réseau AMENDIS_RESEAU (électricité par défaut)
python -m pipeline.steps <étape>         # une seule étape (liste : --list)
python combine_reseaux.py                # après les deux réseaux
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
- Une anomalie est un profil inhabituel par rapport aux contrats de son segment. Ce n'est ni une fraude ni une erreur confirmée. Comme le seuil est de 5 % par segment, chaque segment a environ 5 % de contrats signalés : le taux ne dit pas quel segment est le plus « à risque ».

## Confidentialité

`data/sample/` contient des extraits de données réelles, avec des identifiants hachés. Vérifier avec Amendis que leur publication est autorisée, sinon les retirer du dépôt ou le rendre privé.
