-- Gold : moyenne glissante (3 tours) de la distance à l'or le plus proche.
-- Sert la courbe "convergence vers l'or" du reporting : un bon cerveau fait
-- décroître la distance ; un cerveau bloqué ou perdu stagne/oscille.

select
    r.typologie,
    r.variante,
    t.run_id,
    t.tour,
    t.distance_or,
    avg(t.distance_or) over (
        partition by t.run_id
        order by t.tour
        rows between 2 preceding and current row
    ) as distance_glissante_3
from {{ ref('stg_turns') }} t
join {{ ref('stg_runs') }} r using (run_id)
where t.distance_or is not null
