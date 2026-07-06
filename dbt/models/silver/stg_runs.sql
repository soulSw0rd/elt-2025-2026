-- Silver : runs propres et typés.
-- - dédoublonnage défensif sur run_id (bronze append-only : un re-run accidentel
--   du même fichier ne doit pas fausser les agrégats) ;
-- - `variante` = clé de regroupement du benchmark (cerveau + rayon de vision) ;
-- - COALESCE sur les colonnes optionnelles pour absorber les anciens schémas.

with source as (
    select * from {{ source('bronze', 'runs') }}
),

dedup as (
    select
        *,
        row_number() over (partition by run_id order by horodatage desc) as rn
    from source
)

select
    run_id,
    schema_version,
    horodatage,
    cerveau,
    coalesce(modele, 'aucun')                          as modele,
    rayon_vision,
    cerveau || coalesce('_r' || rayon_vision, '')      as variante,
    typologie,
    seed,
    taille_carte,
    n_or_initial,
    n_ennemis,
    max_turns,
    objectif,
    pas_optimaux,
    success,
    turns,
    ors_ramasses,
    invalides,
    duree_ms,
    latence_moyenne_ms
from dedup
where rn = 1
