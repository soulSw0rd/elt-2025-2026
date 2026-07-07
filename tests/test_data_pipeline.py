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
        # "defi" est volontairement mono-or (succès binaire du challenge LLM)
        or_minimum = 1 if nom == "defi" else 2
        assert int((m == GOLD).sum()) >= or_minimum, nom
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


def _ecrire_run_ancien_schema(tmp_path, version: int, colonnes_absentes: set[str],
                              **surcharges):
    """Écrit un parquet simulant un run d'un ancien contrat (colonnes en moins)."""
    import pyarrow as pa
    import pyarrow.parquet as _pq

    run, _ = _run_et_records(**surcharges)
    donnees = run.model_dump()
    donnees["schema_version"] = version
    for col in colonnes_absentes:
        del donnees[col]
    schema = pa.schema([f for f in datalog.RUN_SCHEMA if f.name not in colonnes_absentes])
    _pq.write_table(
        pa.Table.from_pylist([donnees], schema=schema),
        tmp_path / "runs" / f"run_{run.run_id}.parquet",
    )


def test_evolution_schema_v1_a_v4(tmp_path):
    """Éprouve le mécanisme d'évolution sur les quatre versions du contrat :
    v1 (sans n_or_accessible), v2 (sans iteration/memoire), v3 (sans raison
    dans les turns) et v4 (actuel) s'unionnent via union_by_name ; les
    colonnes manquantes arrivent en NULL (que le silver absorbe par COALESCE)."""
    duckdb = pytest.importorskip("duckdb")
    import pyarrow as pa
    import pyarrow.parquet as _pq

    # Run v4 écrit par le contrat actuel (turns avec colonne raison).
    run_v4, turns_v4 = _run_et_records(n_or_accessible=3, iteration=2, memoire=True)
    ecrire_bronze(run_v4, turns_v4, dossier=tmp_path)
    assert run_v4.schema_version == 4

    # Runs v1 à v3 simulés : contrats d'époque (colonnes postérieures absentes).
    _ecrire_run_ancien_schema(tmp_path, 1, {"n_or_accessible", "iteration", "memoire"})
    _ecrire_run_ancien_schema(tmp_path, 2, {"iteration", "memoire"}, n_or_accessible=3)
    _ecrire_run_ancien_schema(tmp_path, 3, set(), n_or_accessible=3, iteration=1,
                              memoire=False)

    con = duckdb.connect()
    df = con.sql(
        f"select schema_version, n_or_accessible, "
        f"coalesce(n_or_accessible, n_or_initial) as absorbe, "
        f"iteration, coalesce(memoire, false) as memoire_absorbee "
        f"from read_parquet('{tmp_path}/runs/*.parquet', union_by_name=true) "
        f"order by schema_version"
    ).fetchall()
    assert df[0] == (1, None, 3, None, False)  # v1 : tout NULL, absorbé
    assert df[1] == (2, 3, 3, None, False)     # v2 : BFS présent, itératif NULL
    assert df[2] == (3, 3, 3, 1, False)        # v3 : itératif présent
    assert df[3] == (4, 3, 3, 2, True)         # v4 : contrat complet

    # Côté TURNS : un parquet v3 (sans colonne raison) doit s'unionner avec
    # les turns v4 ; raison arrive en NULL pour les anciens tours.
    run_v3t, turns_v3t = _run_et_records()
    schema_turns_v3 = pa.schema([f for f in datalog.TURN_SCHEMA if f.name != "raison"])
    lignes_v3 = []
    for t in turns_v3t:
        d = t.model_dump()
        d["schema_version"] = 3
        del d["raison"]
        lignes_v3.append(d)
    _pq.write_table(
        pa.Table.from_pylist(lignes_v3, schema=schema_turns_v3),
        tmp_path / "turns" / f"turns_{run_v3t.run_id}.parquet",
    )
    df_t = con.sql(
        f"select schema_version, count(*) as n, count(raison) as raisons_non_null "
        f"from read_parquet('{tmp_path}/turns/*.parquet', union_by_name=true) "
        f"group by schema_version order by schema_version"
    ).fetchall()
    versions = {v: (n, raisons) for v, n, raisons in df_t}
    assert versions[3][1] == 0                 # v3 : raison NULL partout


def test_run_id_unique():
    assert nouveau_run_id() != nouveau_run_id()
