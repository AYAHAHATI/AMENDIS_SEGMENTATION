-- ==========================================================================
-- SUIVI DES EXÉCUTIONS (observabilité du pipeline)
-- --------------------------------------------------------------------------
-- Rempli par pipeline/monitoring.py : une ligne par étape exécutée
-- (et par tentative), une ligne par contrôle qualité.
-- Script idempotent (IF NOT EXISTS / OR REPLACE).
-- ==========================================================================

-- Une ligne par exécution d'une étape (et par tentative Airflow)
CREATE TABLE IF NOT EXISTS suivi_executions (
    suivi_id        SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,        -- identifiant de l'exécution Airflow (ou manuel_...)
    origine         TEXT NOT NULL,        -- airflow / manuel
    etape           TEXT NOT NULL,        -- ex. aggregate_2026
    reseau          TEXT,                 -- BASSE TENSION / EAU / vide = étape commune
    tentative       SMALLINT NOT NULL DEFAULT 1,
    debut           TIMESTAMPTZ NOT NULL,
    fin             TIMESTAMPTZ NOT NULL,
    duree_secondes  DOUBLE PRECISION NOT NULL,
    statut          TEXT NOT NULL,        -- succes / echec / arret_qualite
    lignes          BIGINT,               -- volume produit par l'étape
    avertissements  INTEGER,              -- avertissements qualité (étapes de contrôle)
    message         TEXT                  -- message d'erreur éventuel
);

CREATE INDEX IF NOT EXISTS ix_suivi_run ON suivi_executions (run_id);
CREATE INDEX IF NOT EXISTS ix_suivi_debut ON suivi_executions (debut);


-- Une ligne par contrôle qualité exécuté
CREATE TABLE IF NOT EXISTS controles_qualite (
    controle_id     SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    date_execution  TIMESTAMPTZ NOT NULL,
    etape           TEXT NOT NULL,        -- silver_2026, profils_2026, historique_2024...
    reseau          TEXT,
    controle        TEXT NOT NULL,
    severite        TEXT NOT NULL,        -- BLOQUANT / AVERTISSEMENT
    statut          TEXT NOT NULL,        -- OK / ECHEC
    valeur          TEXT,
    regle           TEXT
);

CREATE INDEX IF NOT EXISTS ix_controles_run ON controles_qualite (run_id);


-- Synthèse par exécution : début, fin, durée, nombre d'étapes et d'échecs.
-- Seule la DERNIÈRE tentative de chaque étape compte pour le statut.
CREATE OR REPLACE VIEW v_sante_pipeline AS
WITH derniere_tentative AS (
    SELECT DISTINCT ON (run_id, etape, COALESCE(reseau, ''))
           run_id, origine, etape, reseau, tentative, debut, fin, statut,
           avertissements
    FROM suivi_executions
    ORDER BY run_id, etape, COALESCE(reseau, ''), tentative DESC, debut DESC
)
SELECT
    run_id,
    MIN(origine)                                        AS origine,
    MIN(debut)                                          AS debut,
    MAX(fin)                                            AS fin,
    EXTRACT(EPOCH FROM MAX(fin) - MIN(debut))           AS duree_secondes,
    COUNT(*)                                            AS etapes,
    SUM(CASE WHEN statut = 'succes' THEN 1 ELSE 0 END)  AS etapes_reussies,
    SUM(CASE WHEN statut <> 'succes' THEN 1 ELSE 0 END) AS etapes_en_echec,
    SUM(CASE WHEN tentative > 1 THEN 1 ELSE 0 END)      AS etapes_relancees,
    COALESCE(SUM(avertissements), 0)                    AS avertissements_qualite,
    CASE
        WHEN SUM(CASE WHEN statut = 'arret_qualite' THEN 1 ELSE 0 END) > 0 THEN 'arrêt qualité'
        WHEN SUM(CASE WHEN statut = 'echec' THEN 1 ELSE 0 END) > 0 THEN 'échec'
        ELSE 'succès'
    END                                                 AS statut
FROM derniere_tentative
GROUP BY run_id;
