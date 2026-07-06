-- Gold : KPI par run (une ligne par simulation).
-- Croise le résultat global (stg_runs) et la télémétrie agrégée (stg_turns) :
-- - taux_or_ramasse     : ors ramassés / ors initiaux ;
-- - pas_par_piece       : nombre de pas par pièce ramassée ;
-- - surcout_trajectoire : pas réels / pas optimaux BFS (1.0 = trajet parfait,
--   calculé uniquement sur les runs réussis pour comparer à périmètre égal) ;
-- - deplacements_inutiles, coups_bloques, latence p95.

with turns_agg as (
    select
        run_id,
        count(*) filter (where deplacement_inutile)          as deplacements_inutiles,
        count(*) filter (where bloque)                       as coups_bloques,
        quantile_cont(latence_ms, 0.95)                      as latence_p95_ms,
        avg(distance_or)                                     as distance_or_moyenne
    from {{ ref('stg_turns') }}
    group by run_id
)

select
    r.run_id,
    r.horodatage,
    r.cerveau,
    r.modele,
    r.rayon_vision,
    r.variante,
    r.typologie,
    r.seed,
    r.objectif,
    r.max_turns,
    r.n_or_initial,
    r.n_or_accessible,
    r.pas_optimaux,
    r.success,
    r.turns,
    r.ors_ramasses,
    r.invalides,
    r.latence_moyenne_ms,
    t.deplacements_inutiles,
    t.coups_bloques,
    t.latence_p95_ms,
    t.distance_or_moyenne,
    r.ors_ramasses::double / nullif(r.n_or_initial, 0)       as taux_or_ramasse,
    -- v2 : taux rapporté aux ors ATTEIGNABLES (plus juste si un or est emmuré)
    r.ors_ramasses::double / nullif(r.n_or_accessible, 0)    as taux_or_accessible_ramasse,
    case when r.ors_ramasses > 0
         then r.turns::double / r.ors_ramasses end           as pas_par_piece,
    case when r.success and coalesce(r.pas_optimaux, 0) > 0
         then r.turns::double / r.pas_optimaux end           as surcout_trajectoire
from {{ ref('stg_runs') }} r
left join turns_agg t using (run_id)
