# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv (3.13.3)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

import os
from enum import Enum

# %%
load_dotenv()

# %%
LLM_API_URL = os.environ["LLM_API_URL"]
LLM_API_TOKEN = os.environ["LLM_API_TOKEN"]
MODEL = "google/gemma-4-e2b"

# %%
# LLM_API_URL = os.environ["LMSTUDIO_BASE_URL"]
# LLM_API_TOKEN = os.environ["LM_API_TOKEN"]
# MODEL = "gemma-4-26B"

# %%
client = OpenAI(
    base_url=LLM_API_URL,
    api_key=LLM_API_TOKEN
)

# %% [markdown]
# # Modélisation du monde

# %%
VOID        = 0
PLAYER      = 1
ENNEMY      = 2
GOLD        = 3

SYMBOLS = {VOID: "·", PLAYER: "👤", ENNEMY: "👹", GOLD: "💰"}

# %%
initial_map = np.array([
    [0, 3, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 2, 0, 3], # (1, 1) # (1, 4) # (1, 6)
    [0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3], # (5, 6)
    [0, 0, 0, 0, 0, 0, 0],
])
initial_map


# %% [markdown]
# # Couche de contrat

# %%
class Direction(str, Enum):
    HAUT       = "HAUT"
    BAS        = "BAS"
    GAUCHE     = "GAUCHE"
    DROITE     = "DROITE"


class PlayerDecision(BaseModel):
    direction: Direction


MOVES = {
    "HAUT":     (-1, 0),
    "BAS":      ( 1,  0),
    "GAUCHE":   ( 0,  -1),
    "DROITE":   ( 0,   1),
}


# %% [markdown]
# # Moteur de perception

# %%
def localize(world_map, entity):
    positions = np.argwhere(world_map == entity)
    return positions


# %%
def compute_distances(entities_positions, reference_pos):
    if (len(entities_positions) == 0):
        return np.array([])
    
    v = entities_positions - reference_pos
    distances = np.linalg.norm(v, axis=1)
 
    return np.round(distances, 2)


# %%
def perception(world_map):

    player_position = localize(world_map, PLAYER)
    golds_positions = localize(world_map, GOLD)
    ennemies_positions = localize(world_map, ENNEMY)

    golds_distances = compute_distances(golds_positions, player_position)
    ennemies_distances = compute_distances(ennemies_positions, player_position)

    # percreption directionnelle → delta signé
    nearest_gold_delta = {
        "row": 0,
        "col": 0
    }

    return {
        "ennemies_distances": ennemies_distances.tolist(),
        "ennemies_count": len(ennemies_distances),
        "golds_distances": golds_distances.tolist(),
        "golds_count": len(golds_distances),
        "nearest_gold_delta": nearest_gold_delta,
    }


# %%
def show_map(world_map):
    for row in world_map:
        print("\t".join(SYMBOLS.get(cell, "?") for cell in row))
    print('-----------------------------------------------------')


# %% [markdown]
# # Moteur de déplacement

# %%
def allowed_move(world_map: np.ndarray, pos):
    n_rows, n_cols = world_map.shape
    r, c = pos

    if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
        return False
    
    # Retourne False même si c'est GOLD
    return world_map[r, c] == VOID


# %%
def move(world_map: np.ndarray, old_pos, new_pos):
    if not allowed_move(world_map, new_pos):
        return old_pos
    
    entity = world_map[old_pos[0], old_pos[1]]
    world_map[old_pos[0], old_pos[1]] = VOID
    world_map[new_pos[0], new_pos[1]] = entity

    return new_pos


# %% [markdown]
# # Moteur de décision

# %%
def decide(player_perception) -> PlayerDecision | None:
    prompt = f"""
    # Contexte
    - Tu es un joueur qui veut ramasser de l'or

    # Objectif
    - Trouve le plus court chemin vers l'or

    # Perception
    {player_perception}
    """

    # print(prompt)
    print(str(player_perception))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlayerDecision,
        temperature=0
    )

    return response.choices[0].message.parsed or None


# %% [markdown]
# # Game loop (simulation)

# %%
def game_loop(world_map: np.ndarray, max_turns = 10):
    world_map = world_map.copy()
    move_history = []
    
    for turn in range(max_turns):
        print(f"\n =================== [Turn {turn + 1}] ===================")
        show_map(world_map)

        player_pos = localize(world_map, PLAYER)[0]
        
        p = perception(world_map)
        p["move_history"] = move_history

        decision: PlayerDecision | None = decide(p)

        if decision is not None:
            print(f"\t → LLM decision: {decision.direction.value}")

            move_history.append(decision.direction.value)

            d_row, d_col = MOVES[decision.direction.value]
            new_pos = (player_pos[0] + d_row, player_pos[1] + d_col)
            new_pos = move(world_map, player_pos, new_pos)


# %%
game_loop(world_map=initial_map, max_turns=10)

# %% [markdown]
# # ToDo
#
# - Mettre en place le ramassage d'or → Fin de partie
# - Mettre en place la perception directionnelle
