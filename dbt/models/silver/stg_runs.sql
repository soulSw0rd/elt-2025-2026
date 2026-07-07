-- Silver : runs propres et typés.
-- - dédoublonnage défensif sur run_id (bronze append-only : un re-run accidentel
--   du même fichier ne doit pas fausser les agrégats) ;
-- - `variante` = clé de regroupement du benchmark (cerveau + rayon de vision
--   + modèle LLM le cas échéant : l'axe « modèles » est comparable directement) ;
-- - COALESCE sur les colonnes optionnelles pour absorber les anciens schémas :
--   n_or_accessible est arrivé au schema_version 2 ; pour les runs v1 on
--   l'approxime par n_or_initial (tous les layouts figés ont l'or accessible) ;
--   iteration/memoire sont arrivés au schema_version 3 (challenge itératif) :
--   NULL pour tous les runs antérieurs = partie hors séquence, sans mémoire.

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
    cerveau
        || coalesce('_r' || rayon_vision, '')
        || coalesce('_' || modele, '')                 as variante,
    typologie,
    seed,
    taille_carte,
    n_or_initial,
    n_ennemis,
    max_turns,
    objectif,
    pas_optimaux,
    coalesce(n_or_accessible, n_or_initial)            as n_or_accessible,
    iteration,
    coalesce(memoire, false)                           as memoire,
    success,
    turns,
    ors_ramasses,
    invalides,
    duree_ms,
    latence_moyenne_ms
from dedup
where rn = 1
