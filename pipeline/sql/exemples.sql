-- ==========================================================================
-- EXEMPLES DE REQUÊTES SUR L'ENTREPÔT AMENDIS
-- --------------------------------------------------------------------------
-- À lancer dans psql :
--   docker compose exec amendis-dwh psql -U amendis -d amendis_dwh -f /dev/stdin < pipeline/sql/exemples.sql
-- ou une par une après :
--   docker compose exec amendis-dwh psql -U amendis -d amendis_dwh
-- ==========================================================================


-- 1. Évolution du nombre de contrats par réseau sur 5 ans
--    (jointure fait -> dimensions réseau et période)
SELECT r.libelle AS reseau, p.annee, COUNT(*) AS contrats
FROM fait_profil_consommation f
JOIN dim_reseau  r ON r.reseau_id  = f.reseau_id
JOIN dim_periode p ON p.periode_id = f.periode_id
GROUP BY r.libelle, p.annee
ORDER BY r.libelle, p.annee;


-- 2. Répartition des contrats par segment en 2026 (parts en %)
SELECT r.libelle AS reseau, s.libelle AS segment, COUNT(*) AS contrats,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY r.libelle), 2) AS part_pct
FROM fait_profil_consommation f
JOIN dim_reseau  r ON r.reseau_id  = f.reseau_id
JOIN dim_segment s ON s.segment_id = f.segment_id
WHERE f.periode_id = 202601 AND f.profil_complet
GROUP BY r.libelle, s.ordre, s.libelle
ORDER BY r.libelle, s.ordre;


-- 3. Taux d'anomalies par segment en 2026
SELECT r.libelle AS reseau, s.libelle AS segment,
       COUNT(f.est_anomalie) AS evalues,
       SUM(CASE WHEN f.est_anomalie THEN 1 ELSE 0 END) AS anomalies,
       ROUND(100.0 * AVG(CASE WHEN f.est_anomalie THEN 1 ELSE 0 END), 2) AS taux_pct
FROM fait_profil_consommation f
JOIN dim_reseau  r ON r.reseau_id  = f.reseau_id
JOIN dim_segment s ON s.segment_id = f.segment_id
WHERE f.periode_id = 202601 AND f.est_anomalie IS NOT NULL
GROUP BY r.libelle, s.ordre, s.libelle
ORDER BY r.libelle, s.ordre;


-- 4. Contrats qui ont changé de segment entre 2025 et 2026
--    (auto-jointure de la table de faits sur deux périodes)
SELECT r.libelle AS reseau, s25.libelle AS segment_2025, s26.libelle AS segment_2026,
       COUNT(*) AS contrats
FROM fait_profil_consommation a
JOIN fait_profil_consommation b ON b.contrat_id = a.contrat_id AND b.periode_id = 202601
JOIN dim_segment s25 ON s25.segment_id = a.segment_id
JOIN dim_segment s26 ON s26.segment_id = b.segment_id
JOIN dim_reseau  r   ON r.reseau_id    = a.reseau_id
WHERE a.periode_id = 202501 AND a.profil_complet AND b.profil_complet
  AND s25.segment_id <> s26.segment_id
GROUP BY r.libelle, s25.ordre, s25.libelle, s26.ordre, s26.libelle
ORDER BY r.libelle, contrats DESC;


-- 5. Performances des modèles de prédiction (test 2024-2025 et 2026)
SELECT r.libelle AS reseau, m.evaluation, m.modele,
       ROUND(m.accuracy::numeric, 4) AS accuracy, ROUND(m.f1_macro::numeric, 4) AS f1_macro
FROM metriques_modeles m JOIN dim_reseau r ON r.reseau_id = m.reseau_id
ORDER BY r.libelle, m.evaluation, m.f1_macro DESC;


-- 6. Historique des chargements de l'entrepôt
SELECT j.date_chargement, r.libelle AS reseau, j.nb_contrats, j.nb_profils,
       j.nb_predictions, j.duree_secondes
FROM journal_chargements j JOIN dim_reseau r ON r.reseau_id = j.reseau_id
ORDER BY j.date_chargement DESC;


-- ==========================================================================
-- SUIVI DES EXÉCUTIONS (pipeline/monitoring.py)
-- ==========================================================================

-- 7. Santé des dernières exécutions
SELECT run_id, debut, ROUND(duree_secondes::numeric, 1) AS duree_s, etapes,
       etapes_en_echec, etapes_relancees, avertissements_qualite, statut
FROM v_sante_pipeline ORDER BY debut DESC LIMIT 10;


-- 8. Durée moyenne et maximale de chaque étape (repérer les étapes qui ralentissent)
SELECT etape, COALESCE(reseau, '-') AS reseau, COUNT(*) AS executions,
       ROUND(AVG(duree_secondes)::numeric, 1) AS duree_moy_s,
       ROUND(MAX(duree_secondes)::numeric, 1) AS duree_max_s
FROM suivi_executions WHERE statut = 'succes'
GROUP BY etape, reseau ORDER BY duree_moy_s DESC;


-- 9. Contrôles qualité en échec ou en avertissement, par exécution
SELECT run_id, etape, reseau, controle, severite, valeur
FROM controles_qualite WHERE statut = 'ECHEC'
ORDER BY date_execution DESC;
