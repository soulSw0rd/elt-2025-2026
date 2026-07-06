-- Silver : télémétrie par tour, propre et enrichie.
-- Enrichissements analytiques (window functions DuckDB) :
-- - distance_or_prec : distance à l'or au tour précédent ;
-- - deplacement_inutile : le coup n'a NI ramassé d'or NI réduit la distance
--   (NULL quand injugeable : décision invalide, 1er tour, plus d'or) ;
-- - bloque : le cerveau a proposé un coup impossible (mur/ennemi/bord).

with source as (
    select * from {{ source('bronze', 'turns') }}
),

dedup as (
    select
        *,
        row_number() over (partition by run_id, tour order by tour) as rn
    from source
),

enrichi as (
    select
        run_id,
        schema_version,
        tour,
        direction,
        latence_ms,
        bouge,
        or_ramasse,
        pos_avant_row,
        pos_avant_col,
        pos_apres_row,
        pos_apres_col,
        distance_or,
        ors_restants,
        lag(distance_or) over (partition by run_id order by tour) as distance_or_prec
    from dedup
    where rn = 1
)

select
    *,
    case
        when direction is null then null            -- décision invalide : comptée à part
        when or_ramasse then false                  -- ramasser n'est jamais inutile
        when distance_or is null then null          -- plus d'or : injugeable
        when distance_or_prec is null then null     -- 1er tour : pas de référence
        else distance_or >= distance_or_prec
    end as deplacement_inutile,
    (direction is not null and not bouge) as bloque
from enrichi
