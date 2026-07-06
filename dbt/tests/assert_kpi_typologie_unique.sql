-- Test singulier : le couple (typologie, variante) doit être unique dans la
-- table de reporting (sinon les graphiques agrégeraient des doublons).
select typologie, variante, count(*) as n
from {{ ref('kpi_typologie') }}
group by typologie, variante
having count(*) > 1
