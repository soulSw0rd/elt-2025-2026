-- Gold : agrégats métiers par typologie de simulation × variante de cerveau.
-- C'est la table de référence du reporting : une "typologie" de simulation
-- (degage, ennemi_chemin, leurre, aleatoire) croisée avec chaque variante
-- (greedy, llm_local_r1_<modele>, ...). Le seed `modeles_llm` apporte le
-- KPI « nombre de paramètres du modèle » (NULL documenté quand l'éditeur ne
-- le publie pas ; 0 pour la baseline algorithmique).

with agg as (
    select
        typologie,
        variante,
        cerveau,
        rayon_vision,
        modele,
        count(*)                                as n_runs,
        avg(success::int)                       as taux_reussite,
        avg(taux_or_ramasse)                    as taux_or_ramasse,
        avg(taux_or_accessible_ramasse)         as taux_or_accessible_ramasse,
        avg(pas_par_piece)                      as pas_par_piece_moyen,
        avg(surcout_trajectoire)                as surcout_trajectoire_moyen,
        avg(deplacements_inutiles)              as deplacements_inutiles_moyens,
        avg(coups_bloques)                      as coups_bloques_moyens,
        avg(invalides)                          as invalides_moyens,
        avg(latence_moyenne_ms)                 as latence_moyenne_ms,
        max(latence_p95_ms)                     as latence_p95_ms
    from {{ ref('kpi_run') }}
    group by typologie, variante, cerveau, rayon_vision, modele
)

select
    agg.*,
    m.editeur,
    m.n_parametres_mds,
    m.classe_taille
from agg
left join {{ ref('modeles_llm') }} m using (modele)
