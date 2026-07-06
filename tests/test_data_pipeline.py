"""Tests du volet data : perception locale, arbitre BFS, contrat datalog, bronze.

Le moteur est chargé depuis le notebook via `test_npc_brain` (mêmes définitions).
Aucun appel LLM ni dbt ici : on teste le contrat de données et l'écriture bronze.
"""

from functools import partial

import numpy as np
import pyarrow.parquet as pq
import pytest

from test_npc_brain import NS

import datalog
from datalog import (
    SCHEMA_VERSION,
    RunRecord,
    TurnRecord,
    ecrire_bronze,
    nouveau_run_id,
    records_depuis_resultat,
)

VOID, PLAYER, ENNEMY, GOLD = NS["VOID"], NS["PLAYER"], NS["ENNEMY"], NS["GOLD"]
perception_locale = NS["perception_locale"]
pas_optimaux = NS["pas_optimaux"]
LAYOUTS = NS["LAYOUTS"]
initial_map = NS["initial_map"]
decide_greedy = NS["decide_greedy"]
game_loop_suivi = NS["game_loop_suivi"]
PlayerDecision = NS["PlayerDecision"]
Direction = NS["Direction"]


# --------------------------------------------------------------------------- #
# Perception locale (vision limitée)
# --------------------------------------------------------------------------- #
def test_perception_locale_fenetre_rayon_1():
    p = perception_locale(initial_map, rayon=1)
    # Joueur en (1,1), or en (1,0) -> visible dans la fenêtre 3x3
    assert p["vision"] == ["...", "$J.", "..."]
    assert p["position"] == {"row": 1, "col": 1}
    assert p["or_en_vue"] is True
    assert p["rayon"] == 1
    assert p["taille_carte"] == {"rows": 7, "cols": 7}
    assert p["ors_restants"] == 3


def test_perception_locale_bords_hors_grille():
    # Fenêtre rayon 2 autour de (1,1) : la ligne/colonne -1 est hors grille -> '#'
    p = perception_locale(initial_map, rayon=2)
    assert p["vision"][0] == "#####"
    assert all(ligne.startswith("#") for ligne in p["vision"])


def test_perception_locale_sans_delta_directionnel():
    """La vision locale ne pré-mâche AUCUNE direction (équilibre algo/LLM)."""
    p = perception_locale(initial_map, rayon=1)
    assert "nearest_gold_delta" not in p
    assert "golds_distances" not in p


def test_perception_locale_or_hors_champ():
    m = np.array([
        [PLAYER, VOID, VOID, VOID],
        [VOID, VOID, VOID, VOID],
        [VOID, VOID, VOID, GOLD],
        [VOID, VOID, VOID, VOID],
    ])
    p = perception_locale(m, rayon=1)
    assert p["or_en_vue"] is False
    assert p["ors_restants"] == 1


# --------------------------------------------------------------------------- #
# Arbitre BFS (pas optimaux)
# --------------------------------------------------------------------------- #
def test_pas_optimaux_layouts_connus():
    assert pas_optimaux(LAYOUTS["degage"], "premier") == 1
    assert pas_optimaux(LAYOUTS["degage"], "tout") == 11
    # ennemi_chemin : le détour rallonge le premier or (7 > 5 en Manhattan pur)
    assert pas_optimaux(LAYOUTS["ennemi_chemin"], "premier") == 7
    assert pas_optimaux(LAYOUTS["leurre"], "tout") == 15


def test_pas_optimaux_contourne_les_ennemis():
    # Sans mur : 2 pas. Avec mur d'ennemis au milieu : détour obligatoire.
    libre = np.array([
        [PLAYER, VOID, GOLD],
        [VOID, VOID, VOID],
        [VOID, VOID, VOID],
    ])
    mur = np.array([
        [PLAYER, ENNEMY, GOLD],
        [VOID, ENNEMY, VOID],
        [VOID, VOID, VOID],
    ])
    assert pas_optimaux(libre, "premier") == 2
    assert pas_optimaux(mur, "premier") == 6


def test_pas_optimaux_or_inaccessible():
    # Le joueur est totalement emmuré : (0,1) et (1,0) sont des ennemis.
    m = np.array([
        [PLAYER, ENNEMY, GOLD],
        [ENNEMY, ENNEMY, VOID],
        [VOID, VOID, VOID],
    ])
    assert pas_optimaux(m, "premier") is None
    assert pas_optimaux(m, "tout") is None


def test_pas_optimaux_sans_or():
    m = np.array([[PLAYER, VOID], [VOID, VOID]])
    assert pas_optimaux(m, "tout") == 0


# --------------------------------------------------------------------------- #
# Layouts (game design figé)
# --------------------------------------------------------------------------- #
def test_layouts_bien_formes():
    for nom, m in LAYOUTS.items():
        assert int((m == PLAYER).sum()) == 1, nom
        assert int((m == GOLD).sum()) >= 2, nom
        # tout l'or doit rester accessible (sinon la typologie est injouable)
        assert pas_optimaux(m, "tout") is not None, nom


def test_layout_leurre_piege_le_greedy():
    """Raison d'être de la typologie : le greedy fonce vers l'or emmuré et se bloque."""
    res = game_loop_suivi(LAYOUTS["leurre"], decide_fn=decide_greedy,
                          max_turns=30, live=False, objectif="tout")
    assert res["success"] is False
    assert res["ors_ramasses"] == 0


# --------------------------------------------------------------------------- #
# Télémétrie enrichie + perception_fn injectable
# --------------------------------------------------------------------------- #
def test_telemetrie_enrichie():
    res = game_loop_suivi(initial_map, decide_fn=decide_greedy, max_turns=10, live=False)
    h0 = res["historique"][0]
    assert h0["pos_avant"] == (1, 1)
    assert h0["pos_apres"] == (1, 0)
    assert h0["or_ramasse"] is True
    assert h0["ors_restants"] == 2
    assert isinstance(h0["distance_or"], int)


def test_game_loop_avec_perception_locale():
    brain = lambda p: PlayerDecision(direction=Direction.GAUCHE)
    res = game_loop_suivi(
        initial_map, decide_fn=brain, live=False,
        perception_fn=partial(perception_locale, rayon=1),
    )
    assert res["success"] is True  # or adjacent à gauche


# --------------------------------------------------------------------------- #
# Contrat datalog + écriture bronze
# --------------------------------------------------------------------------- #
def _run_et_records(**surcharges):
    res = game_loop_suivi(initial_map, decide_fn=decide_greedy, max_turns=20,
                          live=False, objectif="tout")
    defauts = dict(
        cerveau="greedy", typologie="degage", taille_carte=7,
        n_or_initial=3, n_ennemis=1, max_turns=20, objectif="tout",
        pas_optimaux=pas_optimaux(initial_map, "tout"),
    )
    defauts.update(surcharges)
    return records_depuis_resultat(res, **defauts)


def test_records_depuis_resultat():
    run, turns = _run_et_records()
    assert run.schema_version == SCHEMA_VERSION
    assert len(run.run_id) == 32
    assert run.turns == len(turns)
    assert turns[0].run_id == run.run_id
    assert turns[0].tour == 1
    # cohérence : ors ramassés dans les turns == résultat global
    assert sum(t.or_ramasse for t in turns) == run.ors_ramasses


def test_ecrire_bronze_et_relire(tmp_path):
    run, turns = _run_et_records(modele=None, rayon_vision=None, seed=42)
    chemins = ecrire_bronze(run, turns, dossier=tmp_path)

    t_runs = pq.read_table(chemins["runs"])
    t_turns = pq.read_table(chemins["turns"])
    assert t_runs.num_rows == 1
    assert t_turns.num_rows == len(turns)

    ligne = t_runs.to_pylist()[0]
    assert ligne["run_id"] == run.run_id
    assert ligne["seed"] == 42
    assert ligne["modele"] is None          # colonne typée même si NULL
    assert ligne["schema_version"] == SCHEMA_VERSION


def test_bronze_append_only(tmp_path):
    """Deux runs -> deux fichiers distincts, jamais d'écrasement."""
    r1, t1 = _run_et_records()
    r2, t2 = _run_et_records()
    ecrire_bronze(r1, t1, dossier=tmp_path)
    ecrire_bronze(r2, t2, dossier=tmp_path)
    assert len(list((tmp_path / "runs").glob("*.parquet"))) == 2
    assert len(list((tmp_path / "turns").glob("*.parquet"))) == 2


def test_union_duckdb_sur_bronze(tmp_path):
    """Les fichiers bronze de plusieurs runs s'unionnent proprement dans DuckDB."""
    duckdb = pytest.importorskip("duckdb")
    for _ in range(3):
        run, turns = _run_et_records()
        ecrire_bronze(run, turns, dossier=tmp_path)
    con = duckdb.connect()
    n = con.sql(
        f"select count(distinct run_id) from read_parquet('{tmp_path}/runs/*.parquet', union_by_name=true)"
    ).fetchone()[0]
    assert n == 3


def test_evolution_schema_v1_vers_v2(tmp_path):
    """Éprouve le mécanisme d'évolution : un parquet v1 (sans n_or_accessible)
    et un parquet v2 s'unionnent via union_by_name ; la colonne manquante
    arrive en NULL pour le run v1 (que le silver absorbe par COALESCE)."""
    duckdb = pytest.importorskip("duckdb")
    import pyarrow as pa

    # Run v2 écrit par le contrat actuel.
    run_v2, turns_v2 = _run_et_records(n_or_accessible=3)
    ecrire_bronze(run_v2, turns_v2, dossier=tmp_path)
    assert run_v2.schema_version == 2

    # Run v1 simulé : même contrat SANS la colonne v2 (schéma d'époque).
    run_v1, _ = _run_et_records()
    donnees_v1 = run_v1.model_dump()
    donnees_v1["schema_version"] = 1
    del donnees_v1["n_or_accessible"]
    schema_v1 = pa.schema([f for f in datalog.RUN_SCHEMA if f.name != "n_or_accessible"])
    import pyarrow.parquet as _pq
    _pq.write_table(
        pa.Table.from_pylist([donnees_v1], schema=schema_v1),
        tmp_path / "runs" / f"run_{run_v1.run_id}.parquet",
    )

    con = duckdb.connect()
    df = con.sql(
        f"select schema_version, n_or_accessible, "
        f"coalesce(n_or_accessible, n_or_initial) as absorbe "
        f"from read_parquet('{tmp_path}/runs/*.parquet', union_by_name=true) "
        f"order by schema_version"
    ).fetchall()
    assert df[0] == (1, None, 3)   # v1 : NULL, absorbé par n_or_initial
    assert df[1] == (2, 3, 3)      # v2 : valeur mesurée par BFS


def test_run_id_unique():
    assert nouveau_run_id() != nouveau_run_id()
