-- ==========================================================================
-- ENTREPÔT DE DONNÉES AMENDIS (PostgreSQL) — SCHÉMA EN ÉTOILE
-- --------------------------------------------------------------------------
-- Alimenté par pipeline/warehouse.py à partir de la couche GOLD du data lake.
-- Ce script est idempotent : il peut être relancé sans rien casser
-- (CREATE ... IF NOT EXISTS, CREATE OR REPLACE VIEW).
--
--                         dim_periode
--                              |
--   dim_contrat --- fait_profil_consommation --- dim_segment
--                              |
--                         dim_reseau
--
--   fait_prediction_segment : segment prédit / de référence (2026)
--   metriques_modeles       : performances des modèles de prédiction
-- ==========================================================================


-- --------------------------------------------------------------------------
-- DIMENSIONS
-- --------------------------------------------------------------------------

-- Réseau : électricité (kWh) ou eau (m3). Les volumes des deux réseaux
-- ne doivent jamais être additionnés.
CREATE TABLE IF NOT EXISTS dim_reseau (
    reseau_id   SMALLINT PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,      -- electricite / eau
    libelle     TEXT NOT NULL,             -- BASSE TENSION / EAU
    unite       TEXT NOT NULL              -- kWh / m3
);

INSERT INTO dim_reseau (reseau_id, code, libelle, unite) VALUES
    (1, 'electricite', 'BASSE TENSION', 'kWh'),
    (2, 'eau',         'EAU',           'm3')
ON CONFLICT (reseau_id) DO NOTHING;


-- Période : une fenêtre de 4 mois (janvier-avril de chaque année,
-- septembre-décembre de l'année précédant 2026 pour la prédiction).
CREATE TABLE IF NOT EXISTS dim_periode (
    periode_id  INTEGER PRIMARY KEY,       -- ex. 202601 = janvier-avril 2026
    annee       SMALLINT NOT NULL,
    fenetre     TEXT NOT NULL,             -- janvier-avril / septembre-decembre
    date_debut  DATE NOT NULL,
    date_fin    DATE NOT NULL,
    usage       TEXT NOT NULL,             -- apprentissage / application
    libelle     TEXT NOT NULL              -- ex. "Janvier-avril 2026"
);


-- Segment : un cluster K-Means d'un réseau (les numéros de clusters
-- diffèrent d'un réseau à l'autre), plus le "Profil incomplet" (-1).
CREATE TABLE IF NOT EXISTS dim_segment (
    segment_id          SERIAL PRIMARY KEY,
    reseau_id           SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    cluster             SMALLINT NOT NULL,   -- numéro K-Means, -1 = incomplet
    libelle             TEXT NOT NULL,       -- Très faible ... Très forte consommation
    ordre               SMALLINT NOT NULL,   -- 1 = Très faible ... 5 = Très forte, 0 = incomplet
    conso_moyenne_ref   DOUBLE PRECISION,    -- centroïde (consommation moyenne par relevé)
    conso_totale_ref    DOUBLE PRECISION,    -- centroïde (consommation totale sur 4 mois)
    nb_profils_appr     INTEGER,             -- profils d'apprentissage du segment
    UNIQUE (reseau_id, cluster)
);


-- Contrat : identifiant anonymisé (NUM_CTA_HASH, 64 caractères) remplacé
-- dans les faits par une clé de substitution courte (contrat_id).
CREATE TABLE IF NOT EXISTS dim_contrat (
    contrat_id      SERIAL PRIMARY KEY,
    num_cta_hash    TEXT NOT NULL,
    reseau_id       SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    UNIQUE (num_cta_hash, reseau_id)
);


-- --------------------------------------------------------------------------
-- FAITS
-- --------------------------------------------------------------------------

-- Granularité : UN CONTRAT x UNE PÉRIODE (profil janvier-avril).
-- Historique 2022-2025 (apprentissage) et 2026 (application).
CREATE TABLE IF NOT EXISTS fait_profil_consommation (
    contrat_id      INTEGER  NOT NULL REFERENCES dim_contrat (contrat_id),
    periode_id      INTEGER  NOT NULL REFERENCES dim_periode (periode_id),
    reseau_id       SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    segment_id      INTEGER  NOT NULL REFERENCES dim_segment (segment_id),
    conso_totale    DOUBLE PRECISION NOT NULL,
    conso_moyenne   DOUBLE PRECISION NOT NULL,
    conso_max       DOUBLE PRECISION NOT NULL,
    conso_min       DOUBLE PRECISION NOT NULL,
    nb_releves      SMALLINT NOT NULL,
    profil_complet  BOOLEAN  NOT NULL,
    est_anomalie    BOOLEAN,             -- vide = non évalué (profil incomplet)
    score_anomalie  DOUBLE PRECISION,    -- plus il est bas, plus le profil est atypique
    PRIMARY KEY (contrat_id, periode_id)
);

CREATE INDEX IF NOT EXISTS ix_fait_profil_periode ON fait_profil_consommation (periode_id);
CREATE INDEX IF NOT EXISTS ix_fait_profil_reseau  ON fait_profil_consommation (reseau_id);
CREATE INDEX IF NOT EXISTS ix_fait_profil_segment ON fait_profil_consommation (segment_id);


-- Granularité : UN CONTRAT x UNE PÉRIODE PRÉDITE (janvier-avril 2026).
-- Trois rôles de la dimension segment : segment de septembre-décembre
-- (persistance), segment prédit et segment de référence (K-Means 2026).
CREATE TABLE IF NOT EXISTS fait_prediction_segment (
    contrat_id              INTEGER  NOT NULL REFERENCES dim_contrat (contrat_id),
    periode_id              INTEGER  NOT NULL REFERENCES dim_periode (periode_id),
    reseau_id               SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    segment_sep_dec_id      INTEGER  NOT NULL REFERENCES dim_segment (segment_id),
    segment_predit_id       INTEGER  NOT NULL REFERENCES dim_segment (segment_id),
    segment_reference_id    INTEGER  NOT NULL REFERENCES dim_segment (segment_id),
    correct                 BOOLEAN  NOT NULL,
    PRIMARY KEY (contrat_id, periode_id)
);


-- Performances des modèles (test temporel 2024->2025 et évaluation 2026).
CREATE TABLE IF NOT EXISTS metriques_modeles (
    reseau_id           SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    evaluation          TEXT NOT NULL,     -- "Test 2024-2025" / "Application 2026"
    modele              TEXT NOT NULL,
    contrats_evalues    INTEGER,
    accuracy            DOUBLE PRECISION,
    precision_weighted  DOUBLE PRECISION,
    recall_weighted     DOUBLE PRECISION,
    f1_weighted         DOUBLE PRECISION,
    precision_macro     DOUBLE PRECISION,
    recall_macro        DOUBLE PRECISION,
    f1_macro            DOUBLE PRECISION,
    PRIMARY KEY (reseau_id, evaluation, modele)
);


-- Journal des chargements (une ligne par chargement d'un réseau).
CREATE TABLE IF NOT EXISTS journal_chargements (
    chargement_id   SERIAL PRIMARY KEY,
    date_chargement TIMESTAMPTZ NOT NULL DEFAULT now(),
    reseau_id       SMALLINT NOT NULL REFERENCES dim_reseau (reseau_id),
    nb_contrats     INTEGER,
    nb_profils      INTEGER,
    nb_predictions  INTEGER,
    duree_secondes  DOUBLE PRECISION
);


-- --------------------------------------------------------------------------
-- VUES (lecture simple depuis Power BI ou en SQL)
-- --------------------------------------------------------------------------

-- Une ligne par contrat et par période, avec les libellés en clair.
CREATE OR REPLACE VIEW v_profils AS
SELECT
    c.num_cta_hash,
    r.libelle       AS reseau,
    r.unite,
    p.annee,
    p.libelle       AS periode,
    p.usage,
    s.libelle       AS segment,
    s.ordre         AS ordre_segment,
    f.conso_totale,
    f.conso_moyenne,
    f.conso_max,
    f.conso_min,
    f.nb_releves,
    f.profil_complet,
    f.est_anomalie,
    f.score_anomalie
FROM fait_profil_consommation f
JOIN dim_contrat c ON c.contrat_id = f.contrat_id
JOIN dim_reseau  r ON r.reseau_id  = f.reseau_id
JOIN dim_periode p ON p.periode_id = f.periode_id
JOIN dim_segment s ON s.segment_id = f.segment_id;


-- Prédictions 2026 avec les trois segments en clair.
CREATE OR REPLACE VIEW v_predictions_2026 AS
SELECT
    c.num_cta_hash,
    r.libelle    AS reseau,
    sd.libelle   AS segment_sep_dec_2025,
    sp.libelle   AS segment_predit_2026,
    sr.libelle   AS segment_reference_2026,
    f.correct
FROM fait_prediction_segment f
JOIN dim_contrat c  ON c.contrat_id  = f.contrat_id
JOIN dim_reseau  r  ON r.reseau_id   = f.reseau_id
JOIN dim_segment sd ON sd.segment_id = f.segment_sep_dec_id
JOIN dim_segment sp ON sp.segment_id = f.segment_predit_id
JOIN dim_segment sr ON sr.segment_id = f.segment_reference_id;


-- Synthèse par réseau, période et segment (évolution sur 5 ans).
CREATE OR REPLACE VIEW v_synthese_segments AS
SELECT
    r.libelle                          AS reseau,
    r.unite,
    p.annee,
    s.ordre                            AS ordre_segment,
    s.libelle                          AS segment,
    COUNT(*)                           AS nb_contrats,
    SUM(f.conso_totale)                AS conso_totale,
    AVG(f.conso_moyenne)               AS conso_moyenne,
    SUM(CASE WHEN f.est_anomalie THEN 1 ELSE 0 END) AS nb_anomalies,
    COUNT(f.est_anomalie)              AS nb_evalues
FROM fait_profil_consommation f
JOIN dim_reseau  r ON r.reseau_id  = f.reseau_id
JOIN dim_periode p ON p.periode_id = f.periode_id
JOIN dim_segment s ON s.segment_id = f.segment_id
WHERE p.fenetre = 'janvier-avril'
GROUP BY r.libelle, r.unite, p.annee, s.ordre, s.libelle;
