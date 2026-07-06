"""Contrat de sortie des données de simulation + écriture de la couche bronze.

Rôle dans l'architecture médaillon :

    simulation (game_loop_suivi) ──► datalog ──► data/bronze/*.parquet (bruts)
                                                   └─► dbt (silver/gold)

Principes de cohérence quand la simulation évolue (question de la spec) :

- **bronze append-only** : un run = un fichier parquet, jamais réécrit ;
- **contrat pydantic versionné** : chaque ligne porte `schema_version`. Si une
  colonne apparaît/disparaît, on incrémente `SCHEMA_VERSION` et la couche
  silver absorbe les anciennes versions (COALESCE / valeurs par défaut) ;
- **schémas pyarrow explicites** : les types sont stables même quand une
  colonne est entièrement NULL dans un fichier (ex. `modele` pour greedy),
  ce qui garantit l'union sans conflit de tous les parquet dans DuckDB.

Historique des versions du contrat (mécanisme éprouvé en conditions réelles :
le bronze committé contient des runs v1 ET v2, unis par `union_by_name`) :

- v1 : contrat initial ;
- v2 : ajout de `n_or_accessible` (RunRecord) — nombre d'ors réellement
  atteignables par BFS depuis la position de départ. Les runs v1 n'ont pas la
  colonne (NULL à la lecture) ; le silver l'absorbe par
  COALESCE(n_or_accessible, n_or_initial), approximation documentée.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

SCHEMA_VERSION = 2

BRONZE_DIR = Path(__file__).resolve().parents[1] / "data" / "bronze"


class RunRecord(BaseModel):
    """Une ligne par simulation : paramètres (axes du benchmark) + résultat global."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    horodatage: datetime
    # Axes du benchmark
    cerveau: str                     # "greedy" | "llm_local" | ...
    modele: str | None = None        # modèle LLM (None pour un cerveau algorithmique)
    rayon_vision: int | None = None  # None = perception complète (baseline)
    typologie: str                   # "degage" | "ennemi_chemin" | "leurre" | "aleatoire"
    # Paramètres de la partie
    seed: int | None = None
    taille_carte: int
    n_or_initial: int
    n_ennemis: int
    max_turns: int
    objectif: str
    pas_optimaux: int | None = None  # arbitre BFS (None = or inaccessible)
    # v2 : ors atteignables par BFS au départ (None = run v1, avant la colonne)
    n_or_accessible: int | None = None
    # Résultat global
    success: bool
    turns: int
    ors_ramasses: int
    invalides: int
    duree_ms: int
    latence_moyenne_ms: int


class TurnRecord(BaseModel):
    """Une ligne par tour de jeu : télémétrie brute mesurée par l'arbitre."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    tour: int
    direction: str | None            # None = décision invalide
    latence_ms: int
    bouge: bool
    or_ramasse: bool
    pos_avant_row: int
    pos_avant_col: int
    pos_apres_row: int
    pos_apres_col: int
    distance_or: int | None          # distance Manhattan à l'or le plus proche APRÈS le coup
    ors_restants: int


RUN_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
    ("run_id", pa.string()),
    ("horodatage", pa.timestamp("us", tz="UTC")),
    ("cerveau", pa.string()),
    ("modele", pa.string()),
    ("rayon_vision", pa.int32()),
    ("typologie", pa.string()),
    ("seed", pa.int64()),
    ("taille_carte", pa.int32()),
    ("n_or_initial", pa.int32()),
    ("n_ennemis", pa.int32()),
    ("max_turns", pa.int32()),
    ("objectif", pa.string()),
    ("pas_optimaux", pa.int32()),
    ("n_or_accessible", pa.int32()),
    ("success", pa.bool_()),
    ("turns", pa.int32()),
    ("ors_ramasses", pa.int32()),
    ("invalides", pa.int32()),
    ("duree_ms", pa.int64()),
    ("latence_moyenne_ms", pa.int64()),
])

TURN_SCHEMA = pa.schema([
    ("schema_version", pa.int32()),
    ("run_id", pa.string()),
    ("tour", pa.int32()),
    ("direction", pa.string()),
    ("latence_ms", pa.int64()),
    ("bouge", pa.bool_()),
    ("or_ramasse", pa.bool_()),
    ("pos_avant_row", pa.int32()),
    ("pos_avant_col", pa.int32()),
    ("pos_apres_row", pa.int32()),
    ("pos_apres_col", pa.int32()),
    ("distance_or", pa.int32()),
    ("ors_restants", pa.int32()),
])


def nouveau_run_id() -> str:
    return uuid.uuid4().hex


def records_depuis_resultat(
    resultat: dict,
    *,
    run_id: str | None = None,
    cerveau: str,
    typologie: str,
    taille_carte: int,
    n_or_initial: int,
    n_ennemis: int,
    max_turns: int,
    objectif: str,
    modele: str | None = None,
    rayon_vision: int | None = None,
    seed: int | None = None,
    pas_optimaux: int | None = None,
    n_or_accessible: int | None = None,
) -> tuple[RunRecord, list[TurnRecord]]:
    """Convertit un résultat de `game_loop_suivi` en records bronze (le contrat)."""
    run_id = run_id or nouveau_run_id()

    run = RunRecord(
        run_id=run_id,
        horodatage=datetime.now(timezone.utc),
        cerveau=cerveau,
        modele=modele,
        rayon_vision=rayon_vision,
        typologie=typologie,
        seed=seed,
        taille_carte=taille_carte,
        n_or_initial=n_or_initial,
        n_ennemis=n_ennemis,
        max_turns=max_turns,
        objectif=objectif,
        pas_optimaux=pas_optimaux,
        n_or_accessible=n_or_accessible,
        success=resultat["success"],
        turns=resultat["turns"],
        ors_ramasses=resultat["ors_ramasses"],
        invalides=resultat["invalides"],
        duree_ms=resultat["duree_ms"],
        latence_moyenne_ms=resultat["latence_moyenne_ms"],
    )

    turns = [
        TurnRecord(
            run_id=run_id,
            tour=h["tour"],
            direction=h["direction"],
            latence_ms=h["latence_ms"],
            bouge=h["bouge"],
            or_ramasse=h["or_ramasse"],
            pos_avant_row=h["pos_avant"][0],
            pos_avant_col=h["pos_avant"][1],
            pos_apres_row=h["pos_apres"][0],
            pos_apres_col=h["pos_apres"][1],
            distance_or=h["distance_or"],
            ors_restants=h["ors_restants"],
        )
        for h in resultat["historique"]
    ]
    return run, turns


def ecrire_bronze(run: RunRecord, turns: list[TurnRecord], dossier: Path | None = None) -> dict:
    """Écrit un run en couche bronze : 1 fichier parquet runs + 1 fichier turns.

    Append-only : chaque run a ses propres fichiers, on ne réécrit jamais
    l'existant. Renvoie les chemins écrits.
    """
    dossier = Path(dossier) if dossier else BRONZE_DIR
    dossier_runs = dossier / "runs"
    dossier_turns = dossier / "turns"
    dossier_runs.mkdir(parents=True, exist_ok=True)
    dossier_turns.mkdir(parents=True, exist_ok=True)

    chemin_run = dossier_runs / f"run_{run.run_id}.parquet"
    chemin_turns = dossier_turns / f"turns_{run.run_id}.parquet"

    pq.write_table(
        pa.Table.from_pylist([run.model_dump()], schema=RUN_SCHEMA), chemin_run
    )
    pq.write_table(
        pa.Table.from_pylist([t.model_dump() for t in turns], schema=TURN_SCHEMA),
        chemin_turns,
    )
    return {"runs": chemin_run, "turns": chemin_turns}
