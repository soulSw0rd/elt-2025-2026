"""Tests du moteur de jeu de `npc_brain`.

Le moteur (constantes, perception, déplacement, boucle de jeu) est chargé
DIRECTEMENT depuis `npc_brain.ipynb` afin de tester le vrai code et non une
copie. Les cellules qui *lancent* la simulation (`game_loop(world_map=...)`)
sont ignorées car elles déclencheraient un appel LLM réel.

La couche de décision LLM (`decide` OpenAI, `decide_cursor` Cursor) n'est pas
testée en direct (aucun identifiant) : on injecte un cerveau simulé via le
paramètre `decide_fn` de `game_loop`.
"""

import os

import numpy as np
import pytest

# Valeurs factices pour que les cellules d'init (client OpenAI) s'exécutent
# sans erreur lors du chargement du notebook (aucune connexion réseau).
os.environ.setdefault("LLM_API_URL", "http://localhost/v1")
os.environ.setdefault("LLM_API_TOKEN", "test")

from engine import load_engine  # noqa: E402  (après les valeurs d'env factices)

# Le moteur est chargé une seule fois depuis le notebook (loader partagé avec
# visu.py et benchmark.py). `test_data_pipeline` réutilise ce NS via un import.
NS = load_engine()

VOID, PLAYER, ENNEMY, GOLD = NS["VOID"], NS["PLAYER"], NS["ENNEMY"], NS["GOLD"]
localize = NS["localize"]
compute_distances = NS["compute_distances"]
perception = NS["perception"]
allowed_move = NS["allowed_move"]
move = NS["move"]
game_loop = NS["game_loop"]
Direction = NS["Direction"]
PlayerDecision = NS["PlayerDecision"]
MOVES = NS["MOVES"]
initial_map = NS["initial_map"]
decide_greedy = NS["decide_greedy"]
game_loop_suivi = NS["game_loop_suivi"]
comparer = NS["comparer"]
carte_aleatoire = NS["carte_aleatoire"]


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #
def test_localize_trouve_les_positions():
    m = np.array([[VOID, PLAYER], [GOLD, ENNEMY]])
    assert localize(m, PLAYER).tolist() == [[0, 1]]
    assert localize(m, GOLD).tolist() == [[1, 0]]
    assert localize(m, ENNEMY).tolist() == [[1, 1]]


def test_compute_distances_vide():
    assert compute_distances(np.empty((0, 2)), np.array([[0, 0]])).tolist() == []


def test_compute_distances_manhattan():
    ref = np.array([0, 0])
    pts = np.array([[3, 4], [0, 2]])
    # Manhattan : |3|+|4|=7 (et non 5 en euclidien), |0|+|2|=2
    assert compute_distances(pts, ref).tolist() == [7, 2]


def test_perception_compte_entites():
    p = perception(initial_map)
    assert p["golds_count"] == 3
    assert p["ennemies_count"] == 1
    assert len(p["golds_distances"]) == 3
    assert len(p["ennemies_distances"]) == 1


def test_perception_delta_directionnel():
    # Joueur (1,1), or le plus proche en (1,0) -> delta col = -1
    p = perception(initial_map)
    assert p["nearest_gold_delta"] == {"row": 0, "col": -1}


# --------------------------------------------------------------------------- #
# Déplacement
# --------------------------------------------------------------------------- #
def test_allowed_move_regles():
    m = np.array([[VOID, GOLD], [ENNEMY, PLAYER]])
    assert allowed_move(m, (0, 0))        # VOID
    assert allowed_move(m, (0, 1))        # GOLD autorisé (version notebook)
    assert not allowed_move(m, (1, 0))    # ennemi
    assert not allowed_move(m, (-1, 0))   # hors grille
    assert not allowed_move(m, (0, 5))    # hors grille


def test_move_vers_case_vide():
    m = np.array([[PLAYER, VOID]])
    res = move(m, (0, 0), (0, 1))
    assert res["gold_collected"] is False
    assert tuple(res["new_pos"]) == (0, 1)
    assert m[0, 0] == VOID and m[0, 1] == PLAYER


def test_move_ramasse_or():
    m = np.array([[PLAYER, GOLD]])
    res = move(m, (0, 0), (0, 1))
    assert res["gold_collected"] is True
    assert m[0, 1] == PLAYER


def test_move_bloque_par_ennemi():
    m = np.array([[PLAYER, ENNEMY]])
    res = move(m, (0, 0), (0, 1))
    assert tuple(res["new_pos"]) == (0, 0)
    assert m[0, 0] == PLAYER and m[0, 1] == ENNEMY


# --------------------------------------------------------------------------- #
# Couche de contrat
# --------------------------------------------------------------------------- #
def test_moves_mapping():
    assert MOVES["HAUT"] == (-1, 0)
    assert MOVES["BAS"] == (1, 0)
    assert MOVES["GAUCHE"] == (0, -1)
    assert MOVES["DROITE"] == (0, 1)


def test_player_decision_parse():
    d = PlayerDecision.model_validate_json('{"direction": "GAUCHE"}')
    assert d.direction == Direction.GAUCHE


# --------------------------------------------------------------------------- #
# Boucle de jeu (avec cerveau simulé injecté via decide_fn)
# --------------------------------------------------------------------------- #
def test_game_loop_ramasse_or():
    """Joueur en (1,1), or adjacent en (1,0) -> GAUCHE termine la partie."""
    brain = lambda _p: PlayerDecision(direction=Direction.GAUCHE)
    res = game_loop(initial_map, max_turns=5, decide_fn=brain, verbose=False)
    assert res["success"] is True
    assert res["turns"] == 1
    assert res["moves"] == ["GAUCHE"]


def test_game_loop_decision_none_ne_plante_pas():
    appels = {"n": 0}

    def brain(_p):
        appels["n"] += 1
        return None

    res = game_loop(initial_map, max_turns=3, decide_fn=brain, verbose=False)
    assert appels["n"] == 3          # toutes les itérations, sans mouvement ni crash
    assert res["success"] is False
    assert res["moves"] == []


def test_game_loop_retourne_resultat_structure():
    brain = lambda _p: PlayerDecision(direction=Direction.HAUT)
    res = game_loop(initial_map, max_turns=1, decide_fn=brain, verbose=False)
    assert set(res) == {"success", "turns", "moves", "map"}
    assert res["turns"] == 1


def test_game_loop_nentache_pas_la_carte_source():
    """game_loop travaille sur une copie : la carte d'origine reste intacte."""
    avant = initial_map.copy()
    brain = lambda _p: PlayerDecision(direction=Direction.GAUCHE)
    game_loop(initial_map, max_turns=3, decide_fn=brain, verbose=False)
    assert np.array_equal(initial_map, avant)


def test_decide_greedy_suit_le_delta():
    assert decide_greedy({"nearest_gold_delta": {"row": 0, "col": -1}}).direction == Direction.GAUCHE
    assert decide_greedy({"nearest_gold_delta": {"row": 3, "col": -1}}).direction == Direction.BAS


def test_decide_greedy_aucun_or():
    assert decide_greedy({"nearest_gold_delta": {"row": 0, "col": 0}}) is None


def test_greedy_resout_la_carte_initiale():
    res = game_loop(initial_map, max_turns=10, decide_fn=decide_greedy, verbose=False)
    assert res["success"] is True


# --------------------------------------------------------------------------- #
# Suivi en direct + métriques de performance
# --------------------------------------------------------------------------- #
def test_game_loop_suivi_telemetrie():
    res = game_loop_suivi(initial_map, decide_fn=decide_greedy, max_turns=10, live=False)
    assert res["success"] is True
    assert res["turns"] == len(res["historique"])
    assert res["historique"][0]["direction"] == "GAUCHE"
    assert "latence_ms" in res["historique"][0]
    assert res["latence_moyenne_ms"] >= 0
    assert res["invalides"] == 0


def test_game_loop_suivi_ne_touche_pas_la_carte_source():
    avant = initial_map.copy()
    game_loop_suivi(initial_map, decide_fn=decide_greedy, live=False)
    assert np.array_equal(initial_map, avant)


def test_game_loop_suivi_decision_none():
    res = game_loop_suivi(initial_map, decide_fn=lambda _p: None, max_turns=3, live=False)
    assert res["success"] is False
    assert res["invalides"] == 3
    assert res["moves"] == []


def test_game_loop_suivi_objectif_premier_sarrete_au_1er_or():
    res = game_loop_suivi(initial_map, decide_fn=decide_greedy, max_turns=30, live=False)
    assert res["success"] is True
    assert res["ors_ramasses"] == 1  # défaut : on s'arrête au premier or


def test_game_loop_suivi_objectif_tout_ramasse_tout():
    # Carte sans ennemi : le greedy peut enchaîner tous les ors.
    m = carte_aleatoire(taille=6, n_or=3, n_ennemis=0, rng=np.random.default_rng(0))
    res = game_loop_suivi(m, decide_fn=decide_greedy, max_turns=60, live=False, objectif="tout")
    assert res["ors_ramasses"] == 3
    assert res["success"] is True
    assert int((res["map"] == GOLD).sum()) == 0


def test_carte_aleatoire_contenu():
    m = carte_aleatoire(taille=7, n_or=3, n_ennemis=1, rng=np.random.default_rng(0))
    assert int((m == PLAYER).sum()) == 1
    assert int((m == GOLD).sum()) == 3
    assert int((m == ENNEMY).sum()) == 1


def test_comparer_greedy_metrics():
    lignes = comparer({"greedy": decide_greedy}, n_cartes=5, max_turns=25, seed=0)
    assert len(lignes) == 1
    l = lignes[0]
    assert l["cerveau"] == "greedy"
    assert l["cartes"] == 5
    assert 0.0 <= l["taux_reussite"] <= 1.0
    assert l["tours_moyens"] >= 1.0
