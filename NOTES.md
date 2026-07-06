# Notes d'avancement — npc_brain

Carnet technique du projet : où on en est, et **comment ça marche**. Le notebook
`npc_brain.ipynb` est la **source de vérité** (le moteur y est défini ; `src/visu.py`
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
| ✅ | **Game design figé** : plusieurs pièces, pas de combat, 4 typologies de cartes |
| ✅ | **Perception locale** (`perception_locale`) + cerveau LLM à vision limitée (`decide_llm_local`) |
| ✅ | **Arbitre BFS** (`pas_optimaux`) : référence de trajectoire optimale pour les KPI |
| ✅ | **Contrat de données** (`src/datalog.py`) + couche bronze parquet append-only |
| ✅ | **Pipeline médaillon** dbt-duckdb (silver/gold + tests de données) |
| ✅ | **Reporting HTML** par typologie, régénérable (`src/build_report.py`) |
| ✅ | Suite de tests (42) chargée depuis le notebook |
| ⬜ | Ennemis mobiles, axe « modèles LLM » élargi |

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

### Perception locale (`perception_locale`) — équilibre charge algo / charge LLM

La perception complète ci-dessus **pré-mâche** le travail : avec
`nearest_gold_delta`, le cerveau n'a plus qu'à suivre une flèche (c'est
exactement ce que fait `decide_greedy`). Pour redonner une vraie charge
cognitive au LLM (exigence 1.1 de la spec), `perception_locale(world_map, rayon)`
ne fournit que :

```python
{
  "vision": ["...", "$J.", "..."],   # fenêtre (2r+1)² de symboles, joueur au centre
  "rayon": 1,                        # curseur de charge algorithmique du benchmark
  "position": {"row": 1, "col": 1},  # savoir de jeu (pas de la vision)
  "taille_carte": {"rows": 7, "cols": 7},
  "ors_restants": 3,
  "or_en_vue": True,
}
```

Symboles : `J` joueur, `$` or, `E` ennemi, `.` vide, `#` hors-grille.
**Aucune direction pré-calculée** : trouver l'or, contourner les ennemis, longer
les bordures et explorer hors champ sont à la charge du cerveau. Le **rayon de
vision** est le curseur : rayon 1 (3×3) = LLM presque aveugle, rayon 2 (5×5) =
plus d'information algorithmique fournie, décision plus facile.

### Arbitre BFS (`pas_optimaux`)

Référence de trajectoire pour les KPI : BFS 4-directions (ennemis bloquants),
et pour l'objectif `tout`, plus courte tournée sur les permutations d'ors
(exact tant qu'il y a peu d'ors). C'est **l'arbitre du benchmark, pas un
cerveau** : aucune décision de jeu ne s'appuie dessus. Renvoie `None` si de
l'or est inaccessible.

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

### `decide_llm_local` — LLM à vision locale (cerveau du benchmark)
Reçoit la sortie de `perception_locale` (fenêtre + objectifs de jeu dans le
prompt : ramasser les pièces, ne jamais marcher sur un ennemi, ne pas se coincer
contre les bordures, explorer si aucun or n'est visible). Passe par le même
appel CLI `cursor-agent` que `decide_cursor_cli` (helper commun
`_appel_cursor_cli`). C'est la variante mesurée par le benchmark, déclinée par
rayon de vision (`llm_local_r1`, `llm_local_r2`).

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

## 8. Suivi en direct (rendu + démo visuelle)

Le tableau de bord utilise `rich.live.Live`. Deux choix garantissent un rendu
**fluide et sans scintillement** :

- **grille en cases de largeur fixe** (chaque cellule fait 2 colonnes) → les
  emojis restent alignés d'un tour à l'autre ;
- **panneaux à largeur figée** → le cadre ne « saute » pas quand le texte change.

### Parcours de démonstration (greedy puis LLM)

Le README décrit le parcours complet pas-à-pas
([§ Démonstration visuelle](README.md#démonstration-visuelle-greedy--llm)).
Résumé pour une soutenance :

| Étape | Commande | Durée | Prérequis |
|-------|----------|-------|-----------|
| 1. Algorithme | `uv run python src/visu.py --carte aleatoire --objectif tout --garanti --pause 0.3` | ~10 s | `uv sync` |
| 2. Auth LLM | `cursor-agent login` | une fois | compte Cursor |
| 3. LLM (même carte) | `uv run python src/visu.py --cerveau llm_local --rayon 1 --carte aleatoire --seed <SEED> --objectif tout --pause 1` | ~2–5 min | étape 2 |

**À dire à l'oral :**

- Le panneau **Carte** montre la grille entière à l'observateur ; seul le
  cerveau `llm_local` reçoit une fenêtre limitée (non affichée à part).
- `--garanti` garantit une partie gagnable pour le greedy et affiche la **seed**
  à réutiliser pour comparer le LLM sur la même configuration.
- `--carte leurre` montre un échec **volontaire** du greedy (piège game design).
- `benchmark.py` et `build_report.py` produisent des **KPI agrégés**, pas une
  démo live tour par tour — rôle distinct de `visu.py`.

### Autres commandes utiles

```bash
# Greedy, typologie dégagée (toujours réussi)
uv run python src/visu.py --carte degage --objectif tout --pause 0.3

# LLM perception complète (voit toute la carte)
uv run python src/visu.py --cerveau cursor --carte degage --objectif tout --pause 1

# Carte aléatoire reproductible sans --garanti
uv run python src/visu.py --carte aleatoire --seed 100 --objectif tout --pause 0.3
```

---

## 9. Comparaison rapide en notebook (`comparer`)

Historiquement, le notebook fournit une comparaison légère, **indépendante du
pipeline data** (section 14) :

- `comparer(cerveaux, n_cartes, max_turns, seed)` rejoue **les mêmes cartes
  aléatoires** pour chaque cerveau et agrège : **taux de réussite**,
  **tours moyens**, **latence moyenne / décision** ;
- `graphe_comparaison(lignes)` produit les bar charts (matplotlib).

C'est un outil d'exploration en cellule `# [RUN]` ; le **benchmark de référence**
(axes, KPI, reporting par typologie) passe désormais par le pipeline médaillon
décrit en sections 13-14, dont la sortie est `reports/benchmark.html`.

---

## 10. Tests

`tests/test_npc_brain.py` **charge les définitions directement depuis le
notebook** (les cellules d'exécution marquées `# [RUN]` sont ignorées, donc aucun
appel LLM pendant les tests). 25 tests couvrent : perception (localisation,
Manhattan, delta directionnel), déplacements (case vide, ramassage, blocage),
contrat (`MOVES`, `PlayerDecision`), boucle de jeu, `decide_greedy`, et le suivi
(`game_loop_suivi`, objectifs `premier`/`tout`, `carte_aleatoire`, `comparer`).

`tests/test_data_pipeline.py` (17 tests) couvre le volet data : perception
locale (fenêtre, bords `#`, absence de delta pré-calculé), arbitre BFS
(contournement, inaccessibilité), layouts (bien formés, le leurre piège bien le
greedy), télémétrie enrichie, contrat `datalog` (écriture/relecture parquet,
append-only, union DuckDB multi-runs). S'y ajoutent les **tests dbt** exécutés
par `dbt build`/`dbt test`.

```bash
uv run pytest -v
```

---

## 11. Dépendances

`numpy` (grille/maths), `pydantic` (contrats décision + données), `openai` &
`cursor-sdk` (LLM), `python-dotenv` (env), `rich` (dashboard), `matplotlib`
(graphes), `ipykernel` (notebook), `pytest` (tests) — plus le volet data :
`pyarrow` (écriture parquet bronze), `duckdb` (warehouse + lecture gold),
`dbt-duckdb` (pipeline médaillon), `pandas` (dataframes du reporting) — le tout
géré par **uv**.

---

## 12. Game design figé (spec 1.2)

| Question de la spec | Décision | Justification |
|---------------------|----------|---------------|
| UNE ou PLUSIEURS pièces ? | **Plusieurs** (objectif `tout` par défaut) | parties plus longues → plus de télémétrie par run, KPI plus discriminants (`taux_or_ramasse` gradue l'échec) |
| Combats autorisés ? | **Non** | un combat ajouterait un système (PV, dégâts) sans enrichir la question centrale « qualité de décision de déplacement » |
| Combats sur la carte ? | **Non** — les ennemis sont des **obstacles statiques bloquants** | l'ennemi crée le problème intéressant (contournement) sans aléa supplémentaire |
| Layout de départ ? | **4 typologies** (cf. ci-dessous) | chaque typologie isole une difficulté précise |

Typologies (définies dans le notebook, `LAYOUTS`) :

- **`degage`** — 3 ors, aucun ennemi : trajectoire pure, cas favorable à la baseline ;
- **`ennemi_chemin`** — un ennemi barre l'axe direct : teste le contournement
  (le greedy s'y bloque, par construction) ;
- **`leurre`** — l'or le plus proche (au sens Manhattan) est emmuré par des
  ennemis, l'or accessible est plus loin : piège anti-greedy, teste la
  résistance au « plus proche d'abord » ;
- **`aleatoire`** — cartes 9×9 seedées (3 ors, 2 ennemis) : généralisation,
  reproductible par seed.

---

## 13. Benchmark : objectifs & KPI (spec 2.1 / 2.2)

**Objectif** : trouver le **point d'équilibre entre charge algorithmique et
charge LLM**, pas la meilleure qualité absolue.

**Axes** :

1. **Charge cognitive algorithmique** : perception complète + delta (`greedy`,
   tout est pré-calculé) vs vision locale rayon 2 vs rayon 1 (le LLM fait tout).
   Le rayon est LE curseur de l'axe.
2. **Typologie de carte** : `degage` / `ennemi_chemin` / `leurre` / `aleatoire`
   (chaque typologie a sa section de reporting).
3. **Modèle LLM** : paramètre `--model` du runner, loggé en bronze (colonne
   `modele`) — l'axe est ouvert sans exploser le temps de run (~10 s/décision).

**KPI (agrégats métiers, table gold `kpi_typologie`)** :

| KPI | Définition | Ce qu'il démontre |
|-----|------------|-------------------|
| `taux_reussite` | runs avec TOUT l'or ramassé / runs | qualité binaire |
| `taux_or_ramasse` | ors ramassés / ors initiaux | qualité graduée (départage les échecs) |
| `pas_par_piece_moyen` | tours joués / pièces ramassées | efficacité brute |
| `surcout_trajectoire_moyen` | pas réels / pas optimaux BFS (runs réussis) | 1.0 = trajet parfait ; mesure la qualité de navigation à périmètre égal |
| `deplacements_inutiles_moyens` | coups n'ayant ni ramassé ni réduit la distance à l'or | hésitations, allers-retours |
| `coups_bloques_moyens` | coups proposés contre un mur/ennemi/bord | compréhension des règles par le cerveau |
| `invalides_moyens` | décisions non parsables (`None`) | robustesse du contrat de sortie LLM |
| `latence_moyenne_ms` / `latence_p95_ms` | coût par décision | le prix de l'intelligence |
| `distance_glissante_3` (table dédiée) | moyenne glissante (3 tours) de la distance à l'or | convergence : décroît si le cerveau progresse, stagne s'il est perdu |

Lecture attendue : `greedy` est gratuit et optimal en terrain dégagé mais échoue
sur `ennemi_chemin`/`leurre` ; le LLM à vision locale paie ~10 s/décision mais
contourne. Entre rayon 1 et rayon 2, on mesure combien d'information
algorithmique il faut fournir au LLM pour qu'il redevienne fiable — c'est le
point d'équilibre recherché.

---

## 14. Architecture data (spec 2.3)

### Contrat de sortie de la simulation

`game_loop_suivi` produit la télémétrie ; `src/datalog.py` la fige en **contrat
pydantic versionné** (`SCHEMA_VERSION`), sérialisé en parquet avec un **schéma
pyarrow explicite** (types stables même quand une colonne est entièrement NULL).

- **`RunRecord`** (1 ligne/run) : `run_id`, `horodatage`, axes du benchmark
  (`cerveau`, `modele`, `rayon_vision`, `typologie`, `seed`), paramètres
  (`taille_carte`, `n_or_initial`, `n_ennemis`, `max_turns`, `objectif`,
  `pas_optimaux`), résultat (`success`, `turns`, `ors_ramasses`, `invalides`,
  `duree_ms`, `latence_moyenne_ms`).
- **`TurnRecord`** (1 ligne/tour) : `run_id`, `tour`, `direction`, `latence_ms`,
  `bouge`, `or_ramasse`, positions avant/après, `distance_or` (mesurée par
  l'arbitre APRÈS le coup), `ors_restants`.

### Cohérence quand la simulation évolue (question de la spec)

1. **Bronze append-only** : un run = un fichier `data/bronze/{runs,turns}/*_<run_id>.parquet`,
   jamais réécrit — les données historiques restent intactes ;
2. **`schema_version` sur chaque ligne** : toute évolution du contrat incrémente
   la version ; les anciennes lignes restent identifiables ;
3. **`union_by_name=true`** à la lecture DuckDB : les colonnes ajoutées plus tard
   arrivent en `NULL` dans les anciens fichiers, pas d'erreur de schéma ;
4. **la couche silver absorbe** : `COALESCE`/défauts dans `stg_runs`/`stg_turns`
   normalisent les versions successives — le gold ne voit qu'un seul schéma.

### Pipeline médaillon (parquet + DuckDB + dbt-duckdb, imposés par la spec)

```
bronze  data/bronze/*.parquet      bruts, append-only (écrits par src/benchmark.py)
   │    source dbt "bronze" : read_parquet(..., union_by_name=true)
silver  stg_runs, stg_turns        vues DuckDB : typage, dédup (run_id), variante,
   │                               deplacement_inutile / bloque (window functions)
gold    kpi_run, kpi_typologie,    tables DuckDB : KPI par run, agrégats par
        distance_glissante         typologie × variante, moyenne glissante
```

- Warehouse : `data/warehouse.duckdb` (régénérable, non versionné) ;
- **Tests dbt** : `unique`/`not_null` sur `run_id`, `accepted_values` sur
  `cerveau`/`typologie`/`direction`, `relationships` turns→runs, test singulier
  d'unicité `(typologie, variante)` sur la table de reporting ;
- Commandes : `uv run dbt build --project-dir dbt --profiles-dir dbt` (depuis la
  racine : les chemins parquet/duckdb y sont relatifs).

### Reporting (par typologie)

`uv run python src/build_report.py` = `dbt build` puis génération de
`reports/benchmark.html` : un graphe d'**arbitrage global qualité/coût**
(taux d'or ramassé vs latence, échelle log) + une **section par typologie**
(barres de qualité/coût, barres d'efficacité de trajectoire, courbe de
convergence). Pas de temps réel : on relance après chaque lot de runs
(conforme à la spec « dbt run + rebuild dataviz »).

---

## 15. Limites connues & pistes

- **Ennemis statiques** : les animer ajouterait un axe « environnement dynamique ».
- **Axe modèles** : comparer plusieurs `--model` (le bronze et le gold sont déjà
  prêts, colonne `modele`).
- **Coût LLM** : ~10 s/décision via CLI → les lots LLM restent petits ; un
  endpoint local (LM Studio) permettrait des volumes plus grands.
- **`pas_optimaux` en objectif `tout`** : permutations exactes — OK pour ≤ 5 ors,
  à remplacer par une heuristique au-delà.
