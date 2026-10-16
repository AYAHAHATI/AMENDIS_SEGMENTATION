"""
monitoring.py

Suivi des exécutions du pipeline (observabilité).

Chaque étape de pipeline/steps.py enregistre UNE LIGNE dans la table
`suivi_executions` de l'entrepôt : étape, réseau, tentative, début, fin,
durée, statut, volume produit, avertissements qualité, message d'erreur.
Les résultats des contrôles qualité vont dans `controles_qualite`.

Identifiant d'exécution (run_id) :
- dans Airflow : celui de l'exécution du DAG (variable AIRFLOW_CTX_DAG_RUN_ID,
  fournie automatiquement à chaque tâche), ainsi que le numéro de tentative ;
- sans Airflow (run_2026.py) : « manuel_<date> », commun à toutes les étapes.

Journalisation « au mieux » (best effort) : si l'entrepôt est
inaccessible, la ligne est écrite dans un fichier CSV de secours
(data/lake/suivi/) et le pipeline continue. Le suivi ne doit jamais faire
échouer le traitement qu'il observe. À la prochaine écriture réussie, les
lignes du fichier de secours sont recopiées dans l'entrepôt (rattrapage).

Usage :
    python -m pipeline.monitoring           # dernières exécutions
    python -m pipeline.monitoring <run_id>  # détail d'une exécution
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.config import LAKE_DIR

SQL_FILE = Path(__file__).resolve().parent / "sql" / "monitoring.sql"
FALLBACK_DIR = LAKE_DIR / "suivi"

RUN_COLUMNS = [
    "run_id", "origine", "etape", "reseau", "tentative", "debut", "fin",
    "duree_secondes", "statut", "lignes", "avertissements", "message",
]
QUALITY_COLUMNS = [
    "run_id", "date_execution", "etape", "reseau", "controle", "severite",
    "statut", "valeur", "regle",
]


# ==========================================================
# CONTEXTE DE L'EXÉCUTION
# ==========================================================

def run_context():
    """(run_id, origine, tentative) de l'exécution en cours."""
    run_id = os.environ.get("AIRFLOW_CTX_DAG_RUN_ID")
    if run_id:
        return run_id, "airflow", int(os.environ.get("AIRFLOW_CTX_TRY_NUMBER") or 1)
    if not os.environ.get("AMENDIS_RUN_ID"):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        os.environ["AMENDIS_RUN_ID"] = f"manuel_{stamp}"
    return os.environ["AMENDIS_RUN_ID"], "manuel", 1


def now():
    return datetime.now(timezone.utc)


# ==========================================================
# ÉCRITURE (entrepôt, ou fichier de secours)
# ==========================================================

def _pending(table, columns):
    """Lignes restées dans le CSV de secours (écrites pendant une panne)."""
    path = FALLBACK_DIR / f"{table}.csv"
    if not path.exists():
        return path, []
    pending = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = [
        {c: (row.get(c) or None) for c in columns}
        for row in pending.to_dict("records")
    ]
    return path, rows


def _write(table, columns, rows):
    """
    Insère des lignes dans l'entrepôt ; sinon les ajoute au CSV de secours.
    Dès que l'entrepôt est de nouveau joignable, les lignes en attente dans
    le CSV de secours y sont recopiées (rattrapage), puis le CSV est supprimé.
    """
    if not rows:
        return "rien"
    try:
        from pipeline.warehouse import connect

        fallback, pending = _pending(table, columns)
        connection = connect()
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(SQL_FILE.read_text(encoding="utf-8"))
                    placeholders = ", ".join(["%s"] * len(columns))
                    cursor.executemany(
                        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                        [[row.get(c) for c in columns] for row in pending + rows],
                    )
        finally:
            connection.close()
        if pending:
            fallback.unlink()
            print(f"  [suivi] rattrapage : {len(pending)} ligne(s) du fichier de secours "
                  f"recopiée(s) dans {table}")
        return "entrepôt"
    except Exception as error:  # noqa: BLE001 - le suivi ne doit jamais casser le pipeline
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        path = FALLBACK_DIR / f"{table}.csv"
        pd.DataFrame(rows, columns=columns).to_csv(
            path, mode="a", index=False, header=not path.exists(), encoding="utf-8-sig"
        )
        first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
        print(f"  [suivi] entrepôt indisponible ({first_line[:80]}) -> {path}")
        return "fichier"


def record_step(step, network, started, ended, status, rows=None, warnings=None, message=None):
    """Enregistre l'exécution d'une étape dans suivi_executions."""
    run_id, origin, attempt = run_context()
    row = {
        "run_id": run_id,
        "origine": origin,
        "etape": step,
        "reseau": network,
        "tentative": attempt,
        "debut": started,
        "fin": ended,
        "duree_secondes": round((ended - started).total_seconds(), 2),
        "statut": status,
        "lignes": None if rows is None else int(rows),
        "avertissements": None if warnings is None else int(warnings),
        "message": None if message is None else str(message)[:1000],
    }
    destination = _write("suivi_executions", RUN_COLUMNS, [row])
    print(f"  [suivi] {step} : {status} ({row['duree_secondes']} s) -> {destination}")


def record_quality(report_table):
    """Enregistre les résultats d'un rapport qualité dans controles_qualite."""
    if report_table is None or report_table.empty:
        return
    run_id, _, _ = run_context()
    rows = [
        {
            "run_id": run_id,
            "date_execution": row["DATE_EXECUTION"],
            "etape": row["ETAPE"],
            "reseau": row["RESEAU"],
            "controle": row["CONTROLE"],
            "severite": row["SEVERITE"],
            "statut": row["STATUT"],
            "valeur": str(row["VALEUR"]),
            "regle": row["REGLE"],
        }
        for _, row in report_table.iterrows()
    ]
    _write("controles_qualite", QUALITY_COLUMNS, rows)


# ==========================================================
# CONSULTATION
# ==========================================================

def show(run_id=None):
    from pipeline.warehouse import query

    if run_id is None:
        print("\nDERNIÈRES EXÉCUTIONS (v_sante_pipeline)")
        runs = query(
            """SELECT run_id, origine, debut, ROUND(duree_secondes::numeric, 1) AS duree_s,
                      etapes, etapes_reussies AS ok, etapes_en_echec AS echecs,
                      etapes_relancees AS relancees, avertissements_qualite AS avert, statut
               FROM v_sante_pipeline ORDER BY debut DESC LIMIT 10"""
        )
        print(runs.to_string(index=False) if len(runs) else "  (aucune exécution enregistrée)")
        if not len(runs):
            return
        run_id = runs.iloc[0]["run_id"]

    print(f"\nDÉTAIL DE L'EXÉCUTION {run_id}")
    steps = query(
        """SELECT etape, COALESCE(reseau, '-') AS reseau, tentative,
                  to_char(debut, 'HH24:MI:SS') AS debut,
                  ROUND(duree_secondes::numeric, 1) AS duree_s, statut, lignes,
                  avertissements AS avert, LEFT(COALESCE(message, ''), 60) AS message
           FROM suivi_executions WHERE run_id = %s ORDER BY debut, reseau""",
        (run_id,),
    )
    for column in ("lignes", "avert"):
        steps[column] = steps[column].astype("Int64")
    print(steps.to_string(index=False))


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else None)
