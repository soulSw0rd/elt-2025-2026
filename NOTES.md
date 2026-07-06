# Notes d'avancement — npc_brain

Carnet technique du projet : où on en est, et **comment ça marche**. Le notebook
`npc_brain.ipynb` est la **source de vérité** (le moteur y est défini ; `visu.py`
et les tests le rechargent tel quel).

---

## 1. Objectif

Un PNJ (le joueur) évolue sur une grille et doit **ramasser de l'or**. La brique
qui décide du prochain déplacement est **interchangeable** :

- un cerveau **déterministe** (baseline) ;
- un **LLM** (modèle Cursor, ou API compatible OpenAI).

Le but pédagogique : comparer une **logique algorithmique** à une décision **LLM**
(charge cognitive, coût/latence, qualité des trajectoires).

---

## 2. État d'avancement

| Statut | Élément |
|--------|---------|
| ✅ | Modélisation du monde (grille, entités, symboles) |
| ✅ | Perception : distances **Manhattan** + `nearest_gold_delta` (perception directionnelle) |
| ✅ | Moteur de déplacement (bornes, ramassage d'or, blocage) |
| ✅ | Cerveau baseline `decide_greedy` (hors-ligne, déterministe) |
| ✅ | Cerveau `decide_cursor_cli` (CLI `cursor-agent`, sans clé API) + `decide` OpenAI |
| ✅ | Boucle de jeu structurée (`game_loop`) + fin de partie sur ramassage |
| ✅ | **Suivi en direct** (`game_loop_suivi` + tableau de bord `rich`) |
| ✅ | **Métriques & benchmark** (`comparer`, `graphe_comparaison`) |
| ✅ | Objectif configurable : `premier` or vs **`tout`** l'or |
| ✅ | Visualisation terminal `visu.py` (options de carte/objectif/vitesse) |
| ✅ | Suite de tests (25) chargée depuis le notebook |
| ⬜ | Contournement d'obstacles (pathfinding) |
| ⬜ | Pipeline data (médaillon bronze/silver/gold) + reporting |

---

## 3. Modèle du monde

La carte est une matrice **NumPy** d'entiers. Chaque case = une entité :

| Valeur | Entité | Symbole |
|-------:|--------|:-------:|
| `0` | vide (`VOID`) | `·` |
| `1` | joueur (`PLAYER`) | 👤 |
| `2` | ennemi (`ENNEMY`) | 👹 |
| `3` | or (`GOLD`) | 💰 |

```python
initial_map = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [3, 1, 0, 0, 2, 0, 3],   # or, joueur, ..., ennemi, ..., or
    ...
])
```

`carte_aleatoire(taille, n_or, n_ennemis, rng)` génère des cartes de test en
plaçant sans collision 1 joueur, `n_or` ors et `n_ennemis` ennemis.

---

## 4. Perception (ce que « voit » le cerveau)

`perception(world_map)` renvoie un dictionnaire résumant l'état **du point de
vue du joueur**. Trois mécanismes :

**a) Localisation** — `localize` s'appuie sur `np.argwhere` pour retrouver les
coordonnées `(ligne, colonne)` de chaque entité.

**b) Distances de Manhattan** — `compute_distances` calcule la distance **L1**
(`|Δligne| + |Δcolonne|`), et non euclidienne :

```python
distances = np.abs(v).sum(axis=1)   # Manhattan
```

> Pourquoi Manhattan ? Les déplacements sont limités à **4 directions**
> (haut/bas/gauche/droite). Le nombre réel de pas pour atteindre une case est
> donc la somme des écarts en ligne et en colonne — pas la diagonale euclidienne.

**c) Perception directionnelle** — `nearest_gold_delta` donne le **vecteur signé**
vers l'or le plus proche (via `argmin` sur les distances) :

```python
nearest = golds_positions[argmin(golds_distances)]
nearest_gold_delta = {"row": Δligne, "col": Δcolonne}   # ex. {"row": 0, "col": -1}
```

C'est cette information directionnelle qui rend une décision « intelligente »
possible sans donner toute la carte au cerveau.

Sortie complète de `perception` :

```python
{
  "ennemies_distances": [...], "ennemies_count": n,
  "golds_distances":    [...], "golds_count":    n,
  "nearest_gold_delta": {"row": .., "col": ..},
}
```

---

## 5. Déplacements & capacités

**Les 4 déplacements possibles** (`MOVES`) mappent une direction vers un vecteur
`(Δligne, Δcolonne)` :

| Direction | Δligne | Δcolonne |
|-----------|:------:|:--------:|
| `HAUT`    | -1 | 0 |
| `BAS`     | +1 | 0 |
| `GAUCHE`  | 0 | -1 |
| `DROITE`  | 0 | +1 |

**Case atteignable** — `allowed_move` valide un déplacement :

```python
def allowed_move(world_map, pos):
    # hors grille -> interdit
    # sinon autorisé uniquement sur une case VIDE ou OR
    return world_map[r, c] in (VOID, GOLD)
```

- On peut marcher sur du **vide** et sur de **l'or** (pour le ramasser).
- Un **ennemi** (ou un bord de grille) **bloque** : le joueur reste sur place.

**Application du déplacement** — `move` mute la grille et signale un ramassage :

```python
{
  "gold_collected": bool,   # True si la case cible contenait de l'or
  "new_pos": (ligne, col),  # position après coup (inchangée si bloqué)
}
```

Mécanique de ramassage : si la cible est de l'or, le joueur s'y déplace (l'or
disparaît de la grille) et `gold_collected=True`.

---

## 6. Cerveaux (capacités de décision)

Tous respectent le même contrat : ils reçoivent une `perception` et renvoient une
`PlayerDecision` (un `Direction` validé par **pydantic**) ou `None`.

### `decide_greedy` — baseline déterministe
Suit `nearest_gold_delta` en progressant d'abord sur **l'axe où l'écart est le
plus grand** :

```python
if abs(d_row) >= abs(d_col):
    direction = BAS if d_row > 0 else HAUT
else:
    direction = DROITE if d_col > 0 else GAUCHE
```

- **Avantage** : instantané (0 ms), optimal en terrain dégagé.
- **Limite** : ne **contourne pas** les obstacles. Si un ennemi barre l'axe visé,
  il rejoue la même direction, reste **bloqué**, et échoue au bout de `max_turns`.

### `decide_cursor_cli` — modèle Cursor (CLI)
Appelle `cursor-agent` en mode **`ask`** (lecture seule) via un sous-processus.
Réutilise la session `cursor-agent login` → **aucune `CURSOR_API_KEY` requise**.
La réponse texte est parsée par `_extract_json`, qui récupère le **dernier bloc
JSON valide** (robuste au texte parasite autour de la réponse).

- **Avantage** : peut raisonner sur la situation, trajectoires souvent plus courtes.
- **Limite** : **latence ~11 s / décision** (appel réseau), coût.

### `decide` — LLM OpenAI
Utilise `client.beta.chat.completions.parse` avec `response_format=PlayerDecision`
(**structured output**). Nécessite `LLM_API_URL` / `LLM_API_TOKEN` dans `.env`.

`cursor_pret()` détecte si Cursor est exploitable (clé réelle **ou** CLI connecté).

---

## 7. Boucle de jeu

### `game_loop` — version simple
Copie la carte, itère `max_turns` fois : perçoit → décide → déplace, s'arrête au
premier or. Retour structuré : `{success, turns, moves, map}`.

### `game_loop_suivi` — version instrumentée (celle du suivi)
Ajoute :

- **Tableau de bord `rich` en direct** (grille + décision + latence + statut) ;
- **Télémétrie par tour** (`historique` : direction, `latence_ms`, `bouge`) ;
- **Objectif configurable** :
  - `"premier"` : fin au 1er or ramassé (défaut) ;
  - `"tout"` : continue jusqu'à ramasser **tout** l'or (ou `max_turns`).

Retour enrichi :

```python
{
  "success": bool, "turns": int, "moves": [...],
  "ors_ramasses": int, "invalides": int,
  "duree_ms": int, "latence_moyenne_ms": int,
  "historique": [{"tour", "direction", "latence_ms", "bouge"}, ...],
  "map": np.ndarray,
}
```

---

## 8. Suivi en direct (rendu)

Le tableau de bord utilise `rich.live.Live`. Deux choix garantissent un rendu
**fluide et sans scintillement** :

- **grille en cases de largeur fixe** (chaque cellule fait 2 colonnes) → les
  emojis restent alignés d'un tour à l'autre ;
- **panneaux à largeur figée** → le cadre ne « saute » pas quand le texte change.

Depuis le terminal :

```bash
uv run python visu.py --carte aleatoire --seed 100 --objectif tout --pause 0.3
```

---

## 9. Métriques & benchmark

- `comparer(cerveaux, n_cartes, max_turns, seed)` rejoue **les mêmes cartes
  aléatoires** pour chaque cerveau et agrège : **taux de réussite**,
  **tours moyens**, **latence moyenne / décision**.
- `graphe_comparaison(lignes)` produit les bar charts (matplotlib).

**Mesure de référence** (greedy sur 20 cartes vs Cursor sur 3, graphique dans
`assets/perf.png`) :

| Cerveau | Réussite | Tours moyens | Latence / décision |
|---------|:--------:|:------------:|:------------------:|
| greedy  | 0,90 | 5,15 | ~0 ms |
| cursor  | 1,00 | 3,67 | **~11 456 ms** |

Lecture : le greedy est gratuit mais imparfait (se bloque sur les ennemis) ; le
LLM trouve des chemins plus courts mais paie une forte latence. C'est l'arbitrage
central du projet.

---

## 10. Tests

`tests/test_npc_brain.py` **charge les définitions directement depuis le
notebook** (les cellules d'exécution marquées `# [RUN]` sont ignorées, donc aucun
appel LLM pendant les tests). 25 tests couvrent : perception (localisation,
Manhattan, delta directionnel), déplacements (case vide, ramassage, blocage),
contrat (`MOVES`, `PlayerDecision`), boucle de jeu, `decide_greedy`, et le suivi
(`game_loop_suivi`, objectifs `premier`/`tout`, `carte_aleatoire`, `comparer`).

```bash
uv run pytest -v
```

---

## 11. Dépendances

`numpy` (grille/maths), `pydantic` (contrat de décision), `openai` &
`cursor-sdk` (LLM), `python-dotenv` (env), `rich` (dashboard), `matplotlib`
(graphes), `ipykernel` (notebook), `pytest` (tests) — le tout géré par **uv**.

---

## 12. Limites connues & pistes

- **Pathfinding** : `decide_greedy` ne contourne pas les ennemis → ajouter un
  A*/BFS comme baseline plus forte, et comparer au LLM.
- **Ennemis statiques** : ils ne bougent pas encore ; on pourrait les animer.
- **Data engineering** : sérialiser `historique`/métriques en architecture
  médaillon (bronze → silver → gold) pour un reporting propre par typologie de
  simulation.
- **Métrique « tout l'or »** : benchmarker sur `ors_ramasses` plutôt que sur la
  réussite binaire pour départager les cerveaux.
