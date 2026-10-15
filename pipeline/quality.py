"""
quality.py

Contrôles qualité automatiques des données (Data Quality).

Avant de lancer les modèles, le pipeline vérifie que les données sont
saines. Chaque contrôle a une sévérité :

- BLOQUANT      : si le contrôle échoue, le pipeline s'ARRÊTE
                  (exception DataQualityError -> la tâche Airflow échoue
                  et les tâches suivantes ne sont pas exécutées) ;
- AVERTISSEMENT : le pipeline continue, mais l'écart est signalé.

Trois points de contrôle dans le traitement 2026 :

1. quality_silver_2026   (après transform)  : les relevés nettoyés
2. quality_profiles_2026 (après features)   : les profils par contrat
3. quality_results_2026  (après create_final) : les segments obtenus

Chaque exécution produit un rapport :
- data/lake/quality/dernier_<etape>_<reseau>.csv  (dernier rapport)
- data/lake/quality/historique_controles.csv      (tous les rapports,
  une ligne par contrôle et par exécution)

Usage :
    python -m pipeline.quality            # les 3 contrôles 2026 du réseau courant
"""

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.config import (
    ID_COLUMN,
    VOLUME_COLUMN,
    DATE_COLUMN,
    MODEL_FEATURES,
    MIN_RELEVES,
    N_CLUSTERS,
    NETWORK,
    APPLICATION_YEAR,
    SEGMENT_WINDOW,
    INTERMEDIATE_DIR,
    FINAL_DIR,
    QUALITY_DIR,
    QUALITY_CONTRACTS_WARNING,
    QUALITY_CONTRACTS_BLOCKING,
    QUALITY_MIN_MONTH_RATIO,
    QUALITY_INCOMPLETE_WARNING,
    QUALITY_INCOMPLETE_BLOCKING,
    QUALITY_NEGATIVE_WARNING,
    QUALITY_NEGATIVE_BLOCKING,
    QUALITY_SEGMENT_DRIFT_WARNING,
    network_slug,
    window_dates,
)

BLOQUANT = "BLOQUANT"
AVERTISSEMENT = "AVERTISSEMENT"

HISTORY_FILE = QUALITY_DIR / "historique_controles.csv"


class DataQualityError(Exception):
    """Levée quand au moins un contrôle BLOQUANT échoue."""


# ==========================================================
# RAPPORT DE CONTRÔLES
# ==========================================================

class QualityReport:
    """
    Collecte les résultats des contrôles d'une étape, les affiche,
    les enregistre, puis arrête le pipeline si un contrôle bloquant
    a échoué.
    """

    def __init__(self, stage, network=NETWORK):
        self.stage = stage
        self.network = network
        self.run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.rows = []

    def check(self, name, passed, value, rule, severity=BLOQUANT):
        """Enregistre le résultat d'un contrôle."""
        self.rows.append({
            "DATE_EXECUTION": self.run_at,
            "ETAPE": self.stage,
            "RESEAU": self.network,
            "CONTROLE": name,
            "SEVERITE": severity,
            "STATUT": "OK" if passed else "ECHEC",
            "VALEUR": value,
            "REGLE": rule,
        })
        return passed

    def graded(self, name, value, warning, blocking, rule, display=None):
        """
        Contrôle à deux seuils : au-delà de `warning` -> avertissement,
        au-delà de `blocking` -> bloquant.
        """
        display = value if display is None else display
        if value > blocking:
            return self.check(name, False, display, rule, BLOQUANT)
        return self.check(name, value <= warning, display, rule, AVERTISSEMENT)

    @property
    def table(self):
        return pd.DataFrame(self.rows)

    @property
    def blocking_failures(self):
        table = self.table
        return table[(table["STATUT"] == "ECHEC") & (table["SEVERITE"] == BLOQUANT)]

    @property
    def warnings(self):
        table = self.table
        return table[(table["STATUT"] == "ECHEC") & (table["SEVERITE"] == AVERTISSEMENT)]

    def show(self):
        print(f"\nCONTRÔLES QUALITÉ - {self.stage} - {self.network}")
        for row in self.rows:
            symbol = "OK " if row["STATUT"] == "OK" else (
                "!! " if row["SEVERITE"] == BLOQUANT else "?  "
            )
            print(f"  [{symbol}] {row['CONTROLE']:<42} {str(row['VALEUR']):>16}"
                  f"   ({row['REGLE']})")
        print(
            f"  -> {len(self.rows)} contrôles | "
            f"{len(self.blocking_failures)} échec(s) bloquant(s) | "
            f"{len(self.warnings)} avertissement(s)"
        )

    def save(self):
        QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        table = self.table
        latest = QUALITY_DIR / f"dernier_{self.stage}_{network_slug(self.network)}.csv"
        table.to_csv(latest, index=False, encoding="utf-8-sig")
        table.to_csv(
            HISTORY_FILE, mode="a", index=False, encoding="utf-8-sig",
            header=not HISTORY_FILE.exists(),
        )
        print(f"  Rapport : {latest}")

    def finish(self, save=True):
        """Affiche, enregistre et arrête le pipeline en cas d'échec bloquant."""
        self.show()
        if save:
            self.save()
        failures = self.blocking_failures
        if len(failures):
            raise DataQualityError(
                f"{len(failures)} contrôle(s) bloquant(s) en échec ({self.stage}) : "
                + ", ".join(failures["CONTROLE"])
            )
        return self


# ==========================================================
# CONTRÔLES DES RELEVÉS (couche silver)
# ==========================================================

def check_readings(readings, report, start_date, end_date, reference=None):
    """
    Contrôles des relevés nettoyés d'un réseau sur une fenêtre.
    reference : relevés de la même fenêtre l'année précédente (optionnel).
    """
    n = len(readings)
    report.check("Relevés présents", n > 0, f"{n:,}", "au moins 1 relevé")
    if n == 0:
        return report

    missing = int(readings[[ID_COLUMN, VOLUME_COLUMN, DATE_COLUMN]].isna().sum().sum())
    report.check("Aucune valeur manquante (contrat, volume, date)",
                 missing == 0, missing, "= 0")

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    outside = int(((readings[DATE_COLUMN] < start) | (readings[DATE_COLUMN] > end)).sum())
    report.check("Dates dans la fenêtre étudiée", outside == 0, outside,
                 f"{start.date()} -> {end.date()}")

    negative = float((readings[VOLUME_COLUMN] < 0).mean())
    report.graded(
        "Volumes négatifs", negative,
        QUALITY_NEGATIVE_WARNING, QUALITY_NEGATIVE_BLOCKING,
        f"avertissement > {QUALITY_NEGATIVE_WARNING:.0%}, "
        f"bloquant > {QUALITY_NEGATIVE_BLOCKING:.0%}",
        display=f"{negative:.2%}",
    )

    duplicates = int(readings.duplicated([ID_COLUMN, DATE_COLUMN, VOLUME_COLUMN]).sum())
    report.check("Lignes en double (contrat, date, volume)", duplicates == 0,
                 f"{duplicates:,}", "= 0 (additionnées ensuite)", AVERTISSEMENT)

    # Couverture : chaque mois de la fenêtre doit être présent et complet
    months = pd.period_range(start, end, freq="M")
    counts = (
        readings[DATE_COLUMN].dt.to_period("M").value_counts()
        .reindex(months, fill_value=0)
    )
    median = float(np.median(counts.values))
    weak = [str(m) for m, c in counts.items() if c < QUALITY_MIN_MONTH_RATIO * median]
    report.check(
        "Tous les mois de la fenêtre présents", not weak,
        ", ".join(weak) if weak else f"{len(months)} mois",
        f">= {QUALITY_MIN_MONTH_RATIO:.0%} des relevés du mois médian",
    )

    if reference is not None and len(reference):
        current = readings[ID_COLUMN].nunique()
        previous = reference[ID_COLUMN].nunique()
        change = current / previous - 1
        report.graded(
            "Évolution du nombre de contrats / année N-1", abs(change),
            QUALITY_CONTRACTS_WARNING, QUALITY_CONTRACTS_BLOCKING,
            f"{previous:,} -> {current:,} ; avertissement > "
            f"{QUALITY_CONTRACTS_WARNING:.0%}, bloquant > {QUALITY_CONTRACTS_BLOCKING:.0%}",
            display=f"{change:+.1%}",
        )
    return report


# ==========================================================
# CONTRÔLES DES PROFILS (un contrat = une ligne)
# ==========================================================

def check_profiles(profiles, report):
    n = len(profiles)
    report.check("Profils présents", n > 0, f"{n:,}", "au moins 1 profil")
    if n == 0:
        return report

    duplicates = int(profiles[ID_COLUMN].duplicated().sum())
    report.check("Un seul profil par contrat", duplicates == 0, duplicates, "= 0")

    missing = int(profiles[MODEL_FEATURES + ["NB_RELEVES"]].isna().sum().sum())
    report.check("Aucune valeur manquante dans les variables", missing == 0,
                 missing, "= 0")

    tolerance = 1e-6
    order = (
        (profiles["CONSO_MIN"] <= profiles["CONSO_MOYENNE"] + tolerance)
        & (profiles["CONSO_MOYENNE"] <= profiles["CONSO_MAX"] + tolerance)
    )
    report.check("Cohérence MIN <= MOYENNE <= MAX", bool(order.all()),
                 int((~order).sum()), "0 profil incohérent")

    total = np.isclose(
        profiles["CONSO_TOTALE"],
        profiles["CONSO_MOYENNE"] * profiles["NB_RELEVES"],
        rtol=1e-6, atol=1e-6,
    )
    report.check("Cohérence TOTALE = MOYENNE x NB_RELEVES", bool(total.all()),
                 int((~total).sum()), "0 profil incohérent")

    incomplete = float((profiles["NB_RELEVES"] < MIN_RELEVES).mean())
    report.graded(
        "Part de profils incomplets", incomplete,
        QUALITY_INCOMPLETE_WARNING, QUALITY_INCOMPLETE_BLOCKING,
        f"< {MIN_RELEVES} relevés ; avertissement > {QUALITY_INCOMPLETE_WARNING:.0%}, "
        f"bloquant > {QUALITY_INCOMPLETE_BLOCKING:.0%}",
        display=f"{incomplete:.2%}",
    )
    return report


# ==========================================================
# CONTRÔLES DES RÉSULTATS (segments)
# ==========================================================

def check_results(final, report, training=None, expected_rows=None):
    """
    final    : jeu final 2026 (une ligne par contrat, avec LIBELLE_CLUSTER)
    training : profils d'apprentissage segmentés (répartition de référence)
    """
    n = len(final)
    if expected_rows is not None:
        report.check("Tous les profils sont dans le jeu final", n == expected_rows,
                     f"{n:,}", f"= {expected_rows:,} profils")

    unlabelled = int(final["LIBELLE_CLUSTER"].isna().sum())
    report.check("Chaque contrat a un segment", unlabelled == 0, unlabelled, "= 0")

    segmented = final[final["CLUSTER"] >= 0]
    found = segmented["LIBELLE_CLUSTER"].nunique()
    report.check("Tous les segments sont représentés", found == N_CLUSTERS,
                 found, f"= {N_CLUSTERS}", AVERTISSEMENT)

    if training is not None and len(segmented):
        now = segmented["LIBELLE_CLUSTER"].value_counts(normalize=True) * 100
        before = training["LIBELLE_CLUSTER"].value_counts(normalize=True) * 100
        drift = (now.reindex(before.index, fill_value=0) - before).abs()
        worst = drift.idxmax()
        report.check(
            "Répartition des segments proche de l'historique",
            drift.max() <= QUALITY_SEGMENT_DRIFT_WARNING,
            f"{drift.max():.1f} pts",
            f"écart max <= {QUALITY_SEGMENT_DRIFT_WARNING:.0f} pts (pire : {worst})",
            AVERTISSEMENT,
        )
    return report


# ==========================================================
# POINTS DE CONTRÔLE DU TRAITEMENT 2026 (tâches du DAG)
# ==========================================================

def quality_silver_2026():
    """Relevés silver janvier-avril 2026 du réseau, comparés à 2025."""
    from pipeline.lake import read_silver

    start, end = window_dates(APPLICATION_YEAR, SEGMENT_WINDOW)
    readings = read_silver(start_date=start, end_date=end)

    reference = None
    previous_start, previous_end = window_dates(APPLICATION_YEAR - 1, SEGMENT_WINDOW)
    try:
        reference = read_silver(start_date=previous_start, end_date=previous_end)
    except FileNotFoundError:
        print("  (pas d'historique silver : contrôle d'évolution ignoré)")

    report = QualityReport("silver_2026")
    check_readings(readings, report, start, end, reference)
    return report.finish()


def quality_profiles_2026(input_file="features_2026.csv"):
    profiles = pd.read_csv(INTERMEDIATE_DIR / input_file)
    report = QualityReport("profils_2026")
    check_profiles(profiles, report)
    return report.finish()


def quality_results_2026(
    final_file="clients_segmentes_2026.csv",
    profiles_file="features_2026.csv",
):
    final = pd.read_csv(FINAL_DIR / final_file)
    expected = len(pd.read_csv(INTERMEDIATE_DIR / profiles_file, usecols=[ID_COLUMN]))

    training = None
    training_file = INTERMEDIATE_DIR / f"clients_cluster_k{N_CLUSTERS}.csv"
    if training_file.exists():
        training = pd.read_csv(training_file, usecols=["LIBELLE_CLUSTER"])

    report = QualityReport("resultats_2026")
    check_results(final, report, training, expected)
    return report.finish()


def quality_history(history, start_date, end_date):
    """Contrôle global de l'historique silver (toute la période)."""
    report = QualityReport("historique")
    check_readings(history, report, start_date, end_date)
    return report.finish()


def quality_history_by_year(history):
    """
    Contrôle de l'historique ANNÉE PAR ANNÉE, avec les mêmes règles que
    2026 : chaque année est comparée à l'année précédente (évolution du
    nombre de contrats) et doit avoir ses 12 mois de relevés.
    Appelé par main.py avant l'entraînement.
    """
    reports, previous = [], None
    for year in sorted(history[DATE_COLUMN].dt.year.unique()):
        readings = history[history[DATE_COLUMN].dt.year == year]
        report = QualityReport(f"historique_{year}")
        check_readings(readings, report, f"{year}-01-01", f"{year}-12-31", previous)
        reports.append(report.finish())
        previous = readings
    return reports


def quality_training_profiles(profiles):
    """
    Contrôle des profils d'apprentissage janvier-avril, année par année
    (mêmes règles que les profils 2026), avant l'entraînement du K-Means.
    """
    reports, previous = [], None
    for year, part in profiles.groupby("ANNEE"):
        report = QualityReport(f"profils_entrainement_{year}")
        check_profiles(part, report)
        if previous is not None:
            change = len(part) / previous - 1
            report.graded(
                "Évolution du nombre de profils / année N-1", abs(change),
                QUALITY_CONTRACTS_WARNING, QUALITY_CONTRACTS_BLOCKING,
                f"{previous:,} -> {len(part):,} ; avertissement > "
                f"{QUALITY_CONTRACTS_WARNING:.0%}, bloquant > {QUALITY_CONTRACTS_BLOCKING:.0%}",
                display=f"{change:+.1%}",
            )
        reports.append(report.finish())
        previous = len(part)
    return reports


if __name__ == "__main__":
    try:
        quality_silver_2026()
        quality_profiles_2026()
        quality_results_2026()
    except DataQualityError as error:
        print(f"\nPIPELINE ARRÊTÉ : {error}")
        sys.exit(1)
