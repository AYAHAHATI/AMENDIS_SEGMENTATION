"""
warehouse.py

Chargement de l'ENTREPÔT DE DONNÉES (PostgreSQL) à partir de la couche
GOLD du data lake, selon un schéma en étoile (pipeline/sql/schema.sql) :

                        dim_periode
                             |
  dim_contrat --- fait_profil_consommation --- dim_segment
                             |
                        dim_reseau

Démarche (ELT classique) :
1. le schéma est créé s'il n'existe pas (script SQL idempotent) ;
2. les dimensions sont mises à jour (les nouveaux contrats passent par
   une table de STAGING temporaire chargée avec COPY) ;
3. les identifiants longs sont remplacés par les clés de substitution
   des dimensions, puis les faits sont chargés en masse avec COPY
   (commande PostgreSQL très rapide) ;
4. tout se passe dans UNE TRANSACTION : si une étape échoue, rien n'est
   modifié dans l'entrepôt (l'ancienne version reste intacte).

Le chargement d'un réseau remplace toutes ses données (idempotent) :
relancer le chargement donne le même entrepôt, sans doublons.

Usage :
    python -m pipeline.warehouse load       # charge le réseau courant + réconciliation
    python -m pipeline.warehouse reconcile  # compare l'entrepôt au gold
    python -m pipeline.warehouse status     # contenu de l'entrepôt
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

from config.config import (
    DWH_HOST,
    DWH_PORT,
    DWH_DATABASE,
    DWH_USER,
    DWH_PASSWORD,
    ID_COLUMN,
    NETWORK,
    TRAINING_YEARS,
    APPLICATION_YEAR,
    SEGMENT_WINDOW,
    INCOMPLETE_CLUSTER,
    INCOMPLETE_LABEL,
    network_slug,
    window_dates,
)
from pipeline.lake import read_gold

SCHEMA_FILE = Path(__file__).resolve().parent / "sql" / "schema.sql"
MONITORING_FILE = Path(__file__).resolve().parent / "sql" / "monitoring.sql"
RESEAU_IDS = {"electricite": 1, "eau": 2}
MEASURES = ["CONSO_TOTALE", "CONSO_MOYENNE", "CONSO_MAX", "CONSO_MIN"]


# ==========================================================
# CONNEXION
# ==========================================================

def connect():
    """Connexion à l'entrepôt PostgreSQL."""
    try:
        return psycopg2.connect(
            host=DWH_HOST, port=DWH_PORT, dbname=DWH_DATABASE,
            user=DWH_USER, password=DWH_PASSWORD, connect_timeout=10,
        )
    except psycopg2.OperationalError as error:
        raise ConnectionError(
            f"Entrepôt PostgreSQL inaccessible ({DWH_HOST}:{DWH_PORT}). "
            "Vérifier que Docker Desktop est lancé et que le conteneur "
            "tourne : docker compose up -d amendis-dwh\n"
            f"Détail : {error}"
        ) from error


def create_schema(cursor):
    """Crée les tables et les vues (entrepôt + suivi) si elles n'existent pas."""
    cursor.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
    cursor.execute(MONITORING_FILE.read_text(encoding="utf-8"))


def ensure_schema(cursor):
    """
    Crée le schéma seulement s'il n'existe pas encore. Dans le DAG, il est
    créé une fois par la tâche init_dwh : les chargements des deux réseaux,
    qui tournent en parallèle, ne recréent donc pas les vues en même temps.
    """
    cursor.execute("SELECT to_regclass('public.fait_profil_consommation')")
    if cursor.fetchone()[0] is None:
        create_schema(cursor)
        load_periods(cursor)


def init_schema():
    """
    Crée le schéma de l'entrepôt (tables, vues) et la dimension période.
    Exécutée une fois avant les chargements des deux réseaux, qui
    peuvent ensuite tourner en parallèle sans se gêner.
    """
    connection = connect()
    try:
        with connection:
            with connection.cursor() as cursor:
                create_schema(cursor)
                load_periods(cursor)
    finally:
        connection.close()
    print(f"[entrepôt] schéma prêt ({DWH_HOST}:{DWH_PORT}/{DWH_DATABASE})")


def copy_dataframe(cursor, df, table):
    """
    Copie un DataFrame dans une table avec COPY (chargement en masse,
    beaucoup plus rapide que des INSERT ligne par ligne).
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    columns = ", ".join(df.columns)
    cursor.copy_expert(
        f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '')", buffer
    )


def period_id(year, window=SEGMENT_WINDOW):
    """Clé d'une période : 202601 = janvier-avril 2026."""
    return year * 100 + int(window[0][:2])


# ==========================================================
# PRÉPARATION DES DONNÉES (lecture du gold)
# ==========================================================

def prepare_profiles(network):
    """
    Un profil par contrat et par année (janvier-avril), historique
    2022-2025 et 2026, avec son segment et son résultat d'anomalie.
    """
    columns = [ID_COLUMN, "ANNEE"] + MEASURES + ["NB_RELEVES", "PROFIL_COMPLET"]

    history = read_gold("profils_entrainement", network)[columns]
    scored = read_gold("anomalies_historique", network)[
        [ID_COLUMN, "ANNEE", "CLUSTER", "EST_ANOMALIE", "SCORE_ANOMALIE"]
    ]
    history = history.merge(scored, on=[ID_COLUMN, "ANNEE"], how="left",
                            validate="one_to_one")
    # Profils incomplets : pas de segment ni d'évaluation d'anomalie
    history["CLUSTER"] = history["CLUSTER"].fillna(INCOMPLETE_CLUSTER)

    current = read_gold("anomalies_2026", network)
    current["ANNEE"] = APPLICATION_YEAR
    current = current[columns + ["CLUSTER", "EST_ANOMALIE", "SCORE_ANOMALIE"]]

    profiles = pd.concat([history, current], ignore_index=True)
    profiles["CLUSTER"] = profiles["CLUSTER"].astype(int)
    profiles["NB_RELEVES"] = profiles["NB_RELEVES"].astype(int)
    profiles["EST_ANOMALIE"] = profiles["EST_ANOMALIE"].map({1: True, 0: False})
    return profiles


def prepare_segments(network):
    """Une ligne par segment du réseau (+ le profil incomplet)."""
    stats = read_gold("statistiques_segments", network)
    stats = stats.sort_values("CONSO_MOYENNE").reset_index(drop=True)
    segments = pd.DataFrame({
        "cluster": stats["CLUSTER"].astype(int),
        "libelle": stats["LIBELLE_CLUSTER"],
        "ordre": range(1, len(stats) + 1),
        "conso_moyenne_ref": stats["CONSO_MOYENNE"],
        "conso_totale_ref": stats["CONSO_TOTALE"],
        "nb_profils_appr": stats["NB_PROFILS"].astype(int),
    })
    rows = segments.to_dict("records") + [{
        "cluster": INCOMPLETE_CLUSTER, "libelle": INCOMPLETE_LABEL, "ordre": 0,
        "conso_moyenne_ref": None, "conso_totale_ref": None, "nb_profils_appr": None,
    }]
    return pd.DataFrame(rows)


def prepare_metrics(network):
    metric_columns = [
        "MODELE", "ACCURACY", "PRECISION_WEIGHTED", "RECALL_WEIGHTED",
        "F1_WEIGHTED", "PRECISION_MACRO", "RECALL_MACRO", "F1_MACRO",
    ]
    test = read_gold("comparaison_modeles", network)[metric_columns]
    test.insert(0, "evaluation", f"Test {TEST_LABEL}")
    test.insert(1, "contrats_evalues", None)
    applied = read_gold("metriques_prediction_2026", network)
    applied = applied[["CONTRATS_EVALUES"] + metric_columns]
    applied.insert(0, "evaluation", f"Application {APPLICATION_YEAR}")
    applied = applied.rename(columns={"CONTRATS_EVALUES": "contrats_evalues"})
    return pd.concat([test, applied], ignore_index=True)


TEST_LABEL = "2024-2025"


# ==========================================================
# CHARGEMENT
# ==========================================================

def load_network(network=NETWORK):
    """
    Charge (ou recharge) toutes les données d'un réseau dans l'entrepôt,
    en une seule transaction.
    """
    started = time.time()
    slug = network_slug(network)
    reseau_id = RESEAU_IDS[slug]
    print(f"[entrepôt] Chargement du réseau {network} ({DWH_HOST}:{DWH_PORT}/{DWH_DATABASE})")

    profiles = prepare_profiles(network)
    segments = prepare_segments(network)
    predictions = read_gold("predictions_segments_2026", network)
    metrics = prepare_metrics(network)
    print(f"  gold lu : {len(profiles):,} profils, {len(predictions):,} prédictions", flush=True)

    connection = connect()
    try:
        with connection:                       # transaction : tout ou rien
            with connection.cursor() as cursor:
                step("schéma (tables et vues)", ensure_schema, cursor)
                step("dim_segment", load_segments, cursor, reseau_id, segments)
                n_contracts = step("dim_contrat", load_contracts,
                                   cursor, reseau_id, profiles, predictions)
                n_profiles = step(f"fait_profil_consommation ({len(profiles):,} lignes)",
                                  load_profile_facts, cursor, reseau_id, profiles)
                n_predictions = step(f"fait_prediction_segment ({len(predictions):,} lignes)",
                                     load_prediction_facts, cursor, reseau_id, predictions)
                step("metriques_modeles", load_metrics, cursor, reseau_id, metrics)
                print("  validation de la transaction (COMMIT)...", flush=True)
                duration = round(time.time() - started, 1)
                cursor.execute(
                    "INSERT INTO journal_chargements "
                    "(reseau_id, nb_contrats, nb_profils, nb_predictions, duree_secondes) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (reseau_id, n_contracts, n_profiles, n_predictions, duration),
                )
    finally:
        connection.close()

    print(
        f"  chargé en {duration} s : {n_contracts:,} contrats, "
        f"{n_profiles:,} faits de profil, {n_predictions:,} faits de prédiction"
    )
    return {"contrats": n_contracts, "profils": n_profiles, "predictions": n_predictions}


def step(label, function, *args):
    """Exécute une étape du chargement en affichant sa durée."""
    print(f"  {label} ...", end=" ", flush=True)
    started = time.time()
    result = function(*args)
    print(f"{time.time() - started:.1f} s", flush=True)
    return result


def load_periods(cursor):
    """dim_periode : janvier-avril de chaque année d'apprentissage et 2026."""
    for year in TRAINING_YEARS + [APPLICATION_YEAR]:
        start, end = window_dates(year, SEGMENT_WINDOW)
        usage = "application" if year == APPLICATION_YEAR else "apprentissage"
        cursor.execute(
            """INSERT INTO dim_periode
                   (periode_id, annee, fenetre, date_debut, date_fin, usage, libelle)
               VALUES (%s, %s, 'janvier-avril', %s, %s, %s, %s)
               ON CONFLICT (periode_id) DO NOTHING""",
            (period_id(year), year, start, end, usage, f"Janvier-avril {year}"),
        )


def load_segments(cursor, reseau_id, segments):
    """dim_segment : mise à jour des segments du réseau (clé stable)."""
    for row in segments.itertuples(index=False):
        cursor.execute(
            """INSERT INTO dim_segment
                   (reseau_id, cluster, libelle, ordre,
                    conso_moyenne_ref, conso_totale_ref, nb_profils_appr)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (reseau_id, cluster) DO UPDATE SET
                   libelle = EXCLUDED.libelle,
                   ordre = EXCLUDED.ordre,
                   conso_moyenne_ref = EXCLUDED.conso_moyenne_ref,
                   conso_totale_ref = EXCLUDED.conso_totale_ref,
                   nb_profils_appr = EXCLUDED.nb_profils_appr""",
            (reseau_id, int(row.cluster), row.libelle, int(row.ordre),
             none_if_nan(row.conso_moyenne_ref), none_if_nan(row.conso_totale_ref),
             none_if_nan(row.nb_profils_appr, int)),
        )


def load_contracts(cursor, reseau_id, profiles, predictions):
    """dim_contrat : ajoute les contrats nouveaux (les anciens gardent leur clé)."""
    hashes = pd.concat([profiles[ID_COLUMN], predictions[ID_COLUMN]]).drop_duplicates()
    cursor.execute("CREATE TEMP TABLE stg_contrat (num_cta_hash TEXT) ON COMMIT DROP")
    copy_dataframe(cursor, hashes.to_frame("num_cta_hash"), "stg_contrat")
    cursor.execute(
        """INSERT INTO dim_contrat (num_cta_hash, reseau_id)
           SELECT num_cta_hash, %s FROM stg_contrat
           ON CONFLICT (num_cta_hash, reseau_id) DO NOTHING""",
        (reseau_id,),
    )
    cursor.execute("ANALYZE dim_contrat")   # statistiques à jour pour la suite
    return len(hashes)


def key_map(cursor, sql, params, key, value):
    """Lit une correspondance clé naturelle -> clé de substitution."""
    cursor.execute(sql, params)
    return pd.DataFrame(cursor.fetchall(), columns=[key, value])


def contract_keys(cursor, reseau_id):
    return key_map(
        cursor, "SELECT num_cta_hash, contrat_id FROM dim_contrat WHERE reseau_id = %s",
        (reseau_id,), ID_COLUMN, "contrat_id",
    )


def segment_keys(cursor, reseau_id, by):
    return key_map(
        cursor, f"SELECT {by}, segment_id FROM dim_segment WHERE reseau_id = %s",
        (reseau_id,), by, "segment_id",
    )


def resolve(df, keys, on, column=None):
    """
    Remplace une clé naturelle (hash du contrat, numéro ou libellé de
    segment) par la clé de substitution de la dimension. Toute ligne sans
    correspondance arrête le chargement.
    """
    resolved = df.merge(keys, left_on=on, right_on=keys.columns[0], how="left",
                        validate="many_to_one")
    target = column or keys.columns[1]
    if target != keys.columns[1]:
        resolved = resolved.rename(columns={keys.columns[1]: target})
    missing = int(resolved[target].isna().sum())
    if missing:
        raise ValueError(f"{missing} lignes sans correspondance pour {on}.")
    if keys.columns[0] != on and keys.columns[0] in resolved.columns:
        resolved = resolved.drop(columns=keys.columns[0])
    resolved[target] = resolved[target].astype(int)
    return resolved


def load_profile_facts(cursor, reseau_id, profiles):
    """
    fait_profil_consommation : remplace les faits du réseau.

    Les clés de substitution (contrat_id, segment_id) sont résolues en
    Python à partir des dimensions, puis les faits sont chargés en masse
    avec COPY. Cette méthode a un temps d'exécution prévisible, quelle que
    soit la version de PostgreSQL (une jointure SQL sur une table tout juste
    remplie, sans statistiques, peut être très lente).
    """
    cursor.execute("DELETE FROM fait_profil_consommation WHERE reseau_id = %s", (reseau_id,))

    facts = resolve(profiles, contract_keys(cursor, reseau_id), ID_COLUMN)
    facts = resolve(facts, segment_keys(cursor, reseau_id, "cluster"), "CLUSTER")
    facts["periode_id"] = facts["ANNEE"] * 100 + int(SEGMENT_WINDOW[0][:2])
    facts["reseau_id"] = reseau_id
    facts = facts[["contrat_id", "periode_id", "reseau_id", "segment_id"] + MEASURES
                  + ["NB_RELEVES", "PROFIL_COMPLET", "EST_ANOMALIE", "SCORE_ANOMALIE"]]
    facts.columns = [
        "contrat_id", "periode_id", "reseau_id", "segment_id", "conso_totale",
        "conso_moyenne", "conso_max", "conso_min", "nb_releves", "profil_complet",
        "est_anomalie", "score_anomalie",
    ]
    copy_dataframe(cursor, facts, "fait_profil_consommation")
    return len(facts)


def load_prediction_facts(cursor, reseau_id, predictions):
    """fait_prediction_segment : remplace les prédictions du réseau."""
    cursor.execute("DELETE FROM fait_prediction_segment WHERE reseau_id = %s", (reseau_id,))

    labels = segment_keys(cursor, reseau_id, "libelle")
    facts = resolve(predictions, contract_keys(cursor, reseau_id), ID_COLUMN)
    for source, target in [
        ("SEGMENT_SEP_DEC_2025", "segment_sep_dec_id"),
        ("SEGMENT_PREDIT_2026", "segment_predit_id"),
        ("SEGMENT_REFERENCE_2026", "segment_reference_id"),
    ]:
        facts = resolve(facts, labels, source, target)
    facts["periode_id"] = period_id(APPLICATION_YEAR)
    facts["reseau_id"] = reseau_id
    facts = facts[["contrat_id", "periode_id", "reseau_id", "segment_sep_dec_id",
                   "segment_predit_id", "segment_reference_id", "CORRECT"]]
    facts = facts.rename(columns={"CORRECT": "correct"})
    copy_dataframe(cursor, facts, "fait_prediction_segment")
    return len(facts)


def load_metrics(cursor, reseau_id, metrics):
    cursor.execute("DELETE FROM metriques_modeles WHERE reseau_id = %s", (reseau_id,))
    for row in metrics.itertuples(index=False):
        values = list(row)
        cursor.execute(
            """INSERT INTO metriques_modeles
                   (reseau_id, evaluation, contrats_evalues, modele, accuracy,
                    precision_weighted, recall_weighted, f1_weighted,
                    precision_macro, recall_macro, f1_macro)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [reseau_id, values[0], none_if_nan(values[1], int)]
            + [values[2]] + [none_if_nan(v) for v in values[3:]],
        )


def none_if_nan(value, cast=float):
    if value is None or pd.isna(value):
        return None
    return cast(value)


# ==========================================================
# CONTENU DE L'ENTREPÔT
# ==========================================================

STATUS_QUERIES = {
    "Tables": """
        SELECT 'dim_reseau' AS table_, COUNT(*) FROM dim_reseau
        UNION ALL SELECT 'dim_periode', COUNT(*) FROM dim_periode
        UNION ALL SELECT 'dim_segment', COUNT(*) FROM dim_segment
        UNION ALL SELECT 'dim_contrat', COUNT(*) FROM dim_contrat
        UNION ALL SELECT 'fait_profil_consommation', COUNT(*) FROM fait_profil_consommation
        UNION ALL SELECT 'fait_prediction_segment', COUNT(*) FROM fait_prediction_segment
        UNION ALL SELECT 'metriques_modeles', COUNT(*) FROM metriques_modeles
        UNION ALL SELECT 'journal_chargements', COUNT(*) FROM journal_chargements""",
    "Profils par réseau et par année": """
        SELECT r.libelle AS reseau, p.annee, COUNT(*) AS profils,
               SUM(CASE WHEN f.profil_complet THEN 0 ELSE 1 END) AS incomplets,
               SUM(CASE WHEN f.est_anomalie THEN 1 ELSE 0 END) AS anomalies
        FROM fait_profil_consommation f
        JOIN dim_reseau r ON r.reseau_id = f.reseau_id
        JOIN dim_periode p ON p.periode_id = f.periode_id
        GROUP BY r.libelle, p.annee ORDER BY r.libelle, p.annee""",
    "Prédictions 2026": """
        SELECT r.libelle AS reseau, COUNT(*) AS contrats_evalues,
               SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correctes,
               ROUND(100.0 * AVG(CASE WHEN correct THEN 1 ELSE 0 END), 2) AS taux_pct
        FROM fait_prediction_segment f JOIN dim_reseau r ON r.reseau_id = f.reseau_id
        GROUP BY r.libelle ORDER BY r.libelle""",
}


def query(sql, params=None):
    """Exécute une requête de lecture et renvoie un DataFrame."""
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [c.name for c in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        connection.close()


def status():
    for title, sql in STATUS_QUERIES.items():
        print(f"\n{title}")
        print(query(sql).to_string(index=False))


# ==========================================================
# RÉCONCILIATION : l'entrepôt contient-il exactement le gold ?
# ==========================================================

def reconcile(network=NETWORK):
    """
    Compare, année par année, les totaux de l'entrepôt avec ceux du gold :
    nombre de profils, consommation totale, anomalies, prédictions
    correctes. Un écart signifie que des lignes ont été perdues ou
    dupliquées pendant le chargement.
    """
    reseau_id = RESEAU_IDS[network_slug(network)]
    profiles = prepare_profiles(network)
    expected = profiles.groupby("ANNEE").agg(
        profils=(ID_COLUMN, "size"),
        conso_totale=("CONSO_TOTALE", "sum"),
        anomalies=("EST_ANOMALIE", lambda s: int((s == True).sum())),  # noqa: E712
    )
    loaded = query(
        """SELECT p.annee AS "ANNEE", COUNT(*) AS profils,
                  SUM(f.conso_totale) AS conso_totale,
                  SUM(CASE WHEN f.est_anomalie THEN 1 ELSE 0 END) AS anomalies
           FROM fait_profil_consommation f
           JOIN dim_periode p ON p.periode_id = f.periode_id
           WHERE f.reseau_id = %s GROUP BY p.annee""",
        (reseau_id,),
    ).set_index("ANNEE")

    comparison = expected.join(loaded, rsuffix="_entrepot", how="outer")
    ok = (
        (comparison["profils"] == comparison["profils_entrepot"]).all()
        and (comparison["anomalies"] == comparison["anomalies_entrepot"]).all()
        and ((comparison["conso_totale"] - comparison["conso_totale_entrepot"].astype(float))
             .abs() < 1e-6 * comparison["conso_totale"].abs().clip(lower=1)).all()
    )

    predictions = read_gold("predictions_segments_2026", network)
    loaded_predictions = query(
        """SELECT COUNT(*) AS n, SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correctes
           FROM fait_prediction_segment WHERE reseau_id = %s""", (reseau_id,),
    ).iloc[0]
    ok_predictions = (
        int(loaded_predictions["n"]) == len(predictions)
        and int(loaded_predictions["correctes"]) == int(predictions["CORRECT"].sum())
    )

    print(f"\nRÉCONCILIATION gold <-> entrepôt - {network}")
    print(comparison.round(2).to_string())
    print(
        f"  prédictions : gold {len(predictions):,} ({int(predictions['CORRECT'].sum()):,} correctes)"
        f" | entrepôt {int(loaded_predictions['n']):,} ({int(loaded_predictions['correctes']):,} correctes)"
    )
    if not (ok and ok_predictions):
        raise ValueError("Réconciliation en échec : l'entrepôt ne correspond pas au gold.")
    print("  -> OK : l'entrepôt contient exactement les données du gold")
    return comparison


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "load":
        load_network()
        reconcile()
    elif command == "reconcile":
        reconcile()
    elif command == "status":
        status()
    else:
        print(__doc__)
        sys.exit(1)
