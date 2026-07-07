# npc_brain

Jeu sur grille : un PNJ ramasse de l'or en évitant des ennemis. Le **cerveau**
(greedy, LLM cloud, LLM local Ollama) est interchangeable ; chaque partie est
tracée dans un pipeline **bronze → silver → gold** (parquet, DuckDB, dbt) avec
rapport HTML par typologie.

> Guide technique détaillé : [`NOTES.md`](NOTES.md)

## Évaluation rapide (correcteur)

Les **données générées** (`data/bronze/`, `data/warehouse.duckdb`, `reports/`) ne
sont **pas versionnées** : elles se recréent localement en quelques secondes
(greedy uniquement, sans LLM) :

```bash
uv sync
uv run python src/benchmark.py --cerveaux greedy --seeds 8   # bronze (parquet)
uv run python src/build_report.py                            # dbt + reports/benchmark.html
uv run pytest                                              # 43 tests moteur + data
```

Ouvrir ensuite `reports/benchmark.html` dans un navigateur.

## Arborescence

```
.
├── npc_brain.ipynb      # source de vérité : moteur + cerveaux + suivi
├── comparatif_llm.ipynb # challenge LLM locaux : courbes + barres de progression
├── README.md            # guide utilisateur + évaluation correcteur
├── NOTES.md             # justifications techniques (game design, KPI, schéma v1→v4)
├── src/                 # outils Python (chargent le moteur depuis le notebook)
│   ├── engine.py        # loader unique du notebook (partagé visu/benchmark/tests)
│   ├── visu.py          # visualisation en direct dans le terminal
│   ├── benchmark.py     # runner : matrice de simulations -> couche bronze
│   ├── challenge_llm.py # challenge de LLM locaux (Ollama) -> couche bronze
│   ├── datalog.py       # contrat de données (pydantic) + écriture parquet bronze
│   └── build_report.py  # dbt build + rapport HTML par typologie
├── dbt/                 # pipeline médaillon dbt-duckdb (seeds + silver + gold + tests)
├── data/bronze/         # parquet bruts (git-ignoré, généré par benchmark/challenge)
├── reports/             # benchmark.html (git-ignoré, généré par build_report.py)
├── tests/               # pytest : moteur + pipeline de données
├── pyproject.toml       # dépendances (uv)
├── uv.lock              # verrouillage des versions
├── .env.example         # modèle de variables (OpenAI API, optionnel)
└── LICENSE
```

### Inventaire — essentiel vs généré vs retiré

| Catégorie | Fichiers / dossiers | Rôle |
|-----------|---------------------|------|
| **Essentiel (versionné)** | `npc_brain.ipynb`, `comparatif_llm.ipynb`, `src/*.py`, `dbt/`, `tests/`, `README.md`, `NOTES.md`, `pyproject.toml`, `uv.lock`, `.env.example`, `LICENSE`, `data/bronze/*/.gitkeep` | Code, pipeline data, doc, tests — tout ce qu'il faut cloner pour reconstruire le projet. |
| **Généré localement (git-ignoré)** | `data/bronze/runs/*.parquet`, `data/bronze/turns/*.parquet`, `data/warehouse.duckdb`, `reports/benchmark.html`, `dbt/target/`, `dbt/dbt_packages/`, `dbt/logs/`, `.venv/`, `.env` | Recréé par `benchmark.py` / `challenge_llm.py` puis `build_report.py` ; ne pas committer. |
| **Environnement / cache** | `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/` | Artefacts Python ; ignorés automatiquement. |
| **Retiré (obsolète)** | `docs/benchmark_exemple.html`, dossier `docs/`, `PRESENTATION.md` | Rapport HTML figé remplacé par `reports/benchmark.html` ; guide oral retiré (non utilisé). |

## Installation

```bash
uv sync
cp .env.example .env    # uniquement si vous utilisez --cerveau api ou CURSOR_API_KEY
```

Python ≥ 3.13, gestionnaire [`uv`](https://docs.astral.sh/uv/).

## Cerveaux et authentification

| Cerveau | Outil | Auth requise | Perception |
|---------|-------|--------------|------------|
| `greedy` | `visu.py`, `benchmark.py` | aucune | complète |
| `cursor` | `visu.py` | `cursor-agent login` | complète |
| `llm_local` | `visu.py`, `benchmark.py` | `cursor-agent login` | locale (fenêtre 3×3 / 5×5) |
| `api` | `visu.py` | `LLM_API_URL` + `LLM_API_TOKEN` dans `.env` | complète |
| `ollama` | `challenge_llm.py` | Ollama local (`ollama serve`) | locale |

```bash
# Greedy — sans configuration
uv run python src/visu.py --carte degage --objectif tout

# API OpenAI-compatible (ou LM Studio, etc.)
uv run python src/visu.py --cerveau api --carte degage --objectif tout

# Cursor CLI
cursor-agent login
uv run python src/visu.py --cerveau llm_local --rayon 1 --carte defi --objectif tout

# Challenge LLM locaux (axe principal du projet)
uv run python src/challenge_llm.py --carte defi --iterations 5
```

Le notebook `npc_brain.ipynb` expose aussi `decide_cursor` (SDK Cursor, clé
`CURSOR_API_KEY` optionnelle). Voir `.env.example`.

## Démonstration visuelle (greedy → LLM)

Parcours pas-à-pas pour **montrer le jeu en direct dans le terminal** : d'abord
l'algorithme (instantané, sans compte), puis le LLM (lent, compte Cursor requis).
Tout passe par `src/visu.py` et le tableau de bord `rich` (carte, décision,
latence, statut tour par tour).

### Ce que vous verrez à l'écran

Pendant la partie, le terminal affiche deux panneaux :

| Panneau | Contenu |
|---------|---------|
| **Carte** | grille ASCII (joueur, or, ennemis) — **vue observateur** (carte entière, même pour `llm_local`) |
| **Suivi** | tour courant, décision (`HAUT`…), latence (ms), nombre de coups, statut |

En fin de partie : `success`, `tours`, `ors_ramasses`, `latence_moy`, `invalides`,
liste des `coups`.

> **Important :** le dashboard montre toujours la carte **complète** à l'humain.
> Le cerveau `llm_local` ne reçoit qu'une **fenêtre locale** (3×3 ou 5×5) ; cette
> fenêtre n'est pas affichée séparément, seulement passée au modèle en entrée.

### Étape 1 — Greedy (~10 s)

Carte aléatoire **gagnable garantie**, objectif « tout l'or » :

```bash
uv run python src/visu.py --carte aleatoire --objectif tout --garanti --pause 0.3
```

- Le greedy décide **instantanément** (`latence ≈ 0 ms` par tour).
- `--garanti` tire une carte aléatoire jusqu'à ce que le greedy puisse tout
  ramasser, puis la rejoue en live.
- **Notez la seed affichée** en début de run (ex. `seed tirée : 1299013684`) :
  elle sert à rejouer **exactement la même carte** à l'étape 2.

Variante typologie figée (terrain libre, toujours réussie) :

```bash
uv run python src/visu.py --carte degage --objectif tout --pause 0.3
```

Contre-exemple volontaire (greedy **bloqué**, même carte à chaque fois) :

```bash
uv run python src/visu.py --carte leurre --objectif tout --pause 0.3
```

### Étape 2 — LLM Cursor (~2–5 min)

Prérequis : `cursor-agent login`.

Rejouer la seed notée à l'étape 1 avec le cerveau benchmark (`llm_local`,
vision locale rayon 1 = fenêtre 3×3) :

```bash
uv run python src/visu.py --cerveau llm_local --rayon 1 \
  --carte aleatoire --seed <SEED_NOTÉE> --objectif tout --pause 1
```

Exemple concret (remplacer par votre seed) :

```bash
uv run python src/visu.py --cerveau llm_local --rayon 1 \
  --carte aleatoire --seed 1299013684 --objectif tout --pause 1
```

Comparer ensuite sur une typologie **figée** (reproductible sans seed) :

```bash
uv run python src/visu.py --cerveau llm_local --rayon 1 \
  --carte ennemi_chemin --objectif tout --pause 1
```

LLM à **perception complète** (plus lent, voit toute la carte comme le greedy) :

```bash
uv run python src/visu.py --cerveau cursor \
  --carte degage --objectif tout --pause 1
```

### Étape 3 — Rapport agrégé (optionnel)

Le benchmark et le HTML ne remplacent **pas** la démo live ; ils agrègent des
centaines de runs pour comparer greedy vs LLM par typologie :

```bash
uv run python src/build_report.py --skip-dbt   # si warehouse déjà à jour
# → ouvrir reports/benchmark.html
```

Détails : [`src/visu.py`](#srcvisupy--visualisation-live-terminal) · [`NOTES.md` §8](NOTES.md#8-suivi-en-direct-rendu--démo-visuelle).

## Typologies de cartes

Les commandes `--carte` (visu) et `--typologies` (benchmark) s'appuient sur
quatre layouts **figés** plus une génération aléatoire :

| Nom | Nature | Rôle |
|-----|--------|------|
| `initiale` | layout fixe (défaut visu) | carte de départ du notebook |
| `degage` | layout fixe | terrain libre, 3 ors, aucun ennemi |
| `ennemi_chemin` | layout fixe | un ennemi barre l'axe direct joueur → or |
| `leurre` | layout fixe | l'or « proche » (Manhattan) est emmuré ; piège pour le greedy |
| `defi` | layout fixe | **challenge itératif** : 1 or, 2 ennemis, succès binaire, progression mesurable |
| `aleatoire` | générée par seed | joueur, ors et ennemis placés au hasard sans collision |

Pour `--carte aleatoire`, les paramètres `--taille`, `--n-or`, `--n-ennemis` et
`--seed` contrôlent la génération. Sans `--seed`, chaque lancement produit une
carte différente. Avec `--seed N`, la carte est **identique à chaque exécution**
(reproductibilité benchmark). Le flag `--garanti` tire des cartes aléatoires
jusqu'à en trouver une que le greedy gagne, puis la rejoue en live (seed
affichée en sortie).

## Workflow complet

```bash
# 1. Simulations → bronze (parquet)
uv run python src/benchmark.py --cerveaux greedy --seeds 8

# 2. Bronze → silver → gold (dbt) + rapport HTML
uv run python src/build_report.py
# → reports/benchmark.html
```

Un jeu de données bronze **n'est pas versionné** (git-ignoré pour alléger le
dépôt). Relancer `benchmark.py` puis `build_report.py` après un clone. Le
benchmark greedy seul suffit pour reconstruire le pipeline sans LLM.

---

## `src/benchmark.py` — matrice de simulations → bronze

Exécute une matrice **cerveau × rayon × typologie × seed** et écrit un couple
de fichiers parquet par run dans `data/bronze/` (`runs_*.parquet`,
`turns_*.parquet`).

```bash
uv run python src/benchmark.py --dry-run
uv run python src/benchmark.py
uv run python src/benchmark.py --cerveaux greedy --seeds 8
uv run python src/benchmark.py --cerveaux llm_local --rayons 1,2 --typologies degage,ennemi_chemin,leurre
uv run python src/benchmark.py --typologies leurre --cerveaux llm_local --rayons 2 --seeds 1
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--cerveaux` | liste séparée par virgules : `greedy`, `llm_local` | `greedy` |
| `--rayons` | rayons de vision pour `llm_local` uniquement (ex. `1,2` → fenêtres 3×3 et 5×5) | `1,2` |
| `--typologies` | liste parmi `degage`, `ennemi_chemin`, `leurre`, `aleatoire` | les quatre |
| `--seeds` | nombre de seeds pour la typologie `aleatoire` (seeds `0 … N-1`) | `5` |
| `--model` | modèle LLM passé à `cursor-agent` (`llm_local` uniquement) | `claude-4-sonnet` |
| `--max-turns` | plafond de tours par run | `40` |
| `--objectif` | `premier` (s'arrête au 1er or) \| `tout` (ramasse tout l'or) | `tout` |
| `--sortie` | dossier bronze de destination | `data/bronze` |
| `--dry-run` | affiche la matrice de runs sans exécuter | off |

**Comportement de la matrice :**

- Layouts fixes (`degage`, `ennemi_chemin`, `leurre`) : **1 run** par combinaison
  cerveau × rayon (pas de seed).
- `aleatoire` : **1 run par seed** ; cartes 9×9, 3 ors, 2 ennemis (constantes
  internes du script, indépendantes des options `--taille` de `visu.py`).
- `greedy` : perception complète, instantané → on peut multiplier les seeds.
- `llm_local` : ~10 s par décision via `cursor-agent` → peu de seeds, c'est
  voulu (point d'équilibre spec, pas qualité maximale). Nécessite
  `cursor-agent login`.

**Axe « modèles LLM » :** relancer la même matrice avec plusieurs `--model`
(ex. `claude-4-sonnet`, `gpt-5-mini`) : le modèle est loggé en bronze, intégré
à la variante (`llm_local_r1_<modele>`) et joint au seed dbt `modeles_llm`
(éditeur, nombre de paramètres, classe de taille) dans le gold.

```bash
uv run python src/benchmark.py --cerveaux llm_local --rayons 1 \
  --typologies degage,ennemi_chemin --model gpt-5-mini
```

---

## Challenge LLM locaux — `src/challenge_llm.py`

Expérience centrale : petits modèles **Ollama** sur la carte `defi`, avec
mémoire inter-parties et coach à aide graduée. Résultats → bronze →
`comparatif_llm.ipynb`.

```bash
ollama serve &
ollama pull qwen2.5:0.5b && ollama pull llama3.2:1b && ollama pull gemma2:2b

uv run python src/challenge_llm.py --carte defi --iterations 5
uv run jupyter execute comparatif_llm.ipynb
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--models` | modèles Ollama, séparés par virgules | `qwen2.5:0.5b,llama3.2:1b,gemma2:2b` |
| `--carte` | `initiale`, `degage`, `ennemi_chemin`, `leurre`, `defi` | `initiale` |
| `--runs` | mode batch : répétitions indépendantes par modèle | `3` |
| `--iterations` | mode itératif : parties successives avec **mémoire inter-parties** (résumé d'expérience relu depuis bronze et injecté dans le prompt) | off |
| `--sans-memoire` | mode itératif sans expérience injectée (groupe témoin) | off |
| `--rayon` | rayon de vision locale | `1` (fenêtre 3×3) |
| `--max-turns` | plafond de tours par run | `30` |
| `--objectif` | `premier` \| `tout` | `tout` |
| `--sortie` | dossier bronze | `data/bronze` |

En mode itératif, chaque partie est enregistrée dans la couche bronze
(`iteration=n`, `memoire=true/false`) ; le notebook `comparatif_llm.ipynb`
trace la **progression itération par itération** (or ramassé, tours jusqu'au
1er or, coups bloqués) — le graphique s'enrichit à chaque nouvelle partie.
À `temperature=0`, toute amélioration est attribuable à la mémoire injectée.

**Carte `defi` (protocole de progression)** : 1 seul or (succès binaire),
2 ennemis adjacents au chemin direct, or hors de la vision 3×3 initiale.
La mémoire y est un **coach à aide graduée**, reconstruit depuis bronze avant
chaque partie — chaque niveau n'est débloqué que par les échecs précédents :

1. **niveau 0** (itération 1) : aucune mémoire ;
2. **niveau 1** (après 1 échec) : cases déjà visitées + consigne anti-boucle ;
3. **niveau 2** (après 2 échecs sans or) : **coach de l'arbitre** — à chaque
   tour, le prochain coup optimal (BFS depuis la position actuelle) est placé
   en consigne finale du prompt ;
4. **niveau 3** (après un succès) : **rejouer sa victoire** — depuis chaque
   position traversée lors de la partie gagnante, le coup alors joué est
   rappelé (l'aide vient de l'expérience du modèle, plus de l'arbitre).

**Traçage des raisonnements (contrat v4)** : chaque décision LLM est loggée
avec sa `raison` (une phrase renvoyée dans le JSON). Le notebook affiche les
**trajectoires sur la carte** (une couleur par itération) et une **table des
raisons** tour par tour, pour confronter ce que le modèle dit à ce qu'il fait.

```bash
uv run python src/challenge_llm.py --carte defi --iterations 5   # le protocole complet
```

Les runs Ollama s'insèrent dans le pipeline existant : `dbt build` les intègre
(seed `modeles_llm` enrichi avec les paramètres **réels** des modèles locaux :
0.5, 1 et 2 Mds) et ils apparaissent dans le rapport HTML comme des variantes
`ollama_r1_<modele>`.

### Interprétation — attributs, inputs et critères de progression

Documentation détaillée : [NOTES.md §16](NOTES.md#16-challenge-llm-locaux-carte-défi). Résumé :

**Attributs des modèles** (seed dbt `modeles_llm`) :

| Modèle | Éditeur | Paramètres | Latence typique/décision |
|--------|---------|------------|--------------------------|
| qwen2.5:0.5b | Alibaba | 0,5 Md | ~0,7 s |
| llama3.2:1b | Meta | 1 Md | ~1,5 s |
| gemma2:2b | Google | 2 Mds | ~3,5 s |

**Ce que l'IA sait avant de se lancer** (fixe, dans le prompt) : règles du jeu
(légende, priorités, directions), format JSON `{"direction", "raison"}`. Elle ne
connaît **pas** la carte entière (pas de plan, pas de positions d'or).

**Inputs à chaque tour** (`perception_locale`, rayon 1) : fenêtre **3×3**,
position, taille de grille, compteur d'ors restants, 8 derniers coups ; en mode
itératif, résumé d'expérience relu depuis bronze ; aux niveaux 2-3, consigne
finale du coach (voir ci-dessous).

**Critères de « progression »** (mesurés par l'arbitre, pas par l'IA) :

1. **or ramassé** croissant d'une itération à l'autre (critère principal sur `defi`) ;
2. **tours jusqu'au 1er or** décroissants ;
3. **coups bloqués** décroissants ;
4. **invalides** (JSON non respecté) = 0.

À `temperature=0`, seule la mémoire injectée varie entre itérations : toute
amélioration est attribuable au protocole coach, pas au hasard.

### Interprétation — graphiques du notebook `comparatif_llm.ipynb`

Le notebook ne contient que les **visualisations** (courbes, barres, trajectoires,
table des raisons). Lancer :

```bash
uv run python src/challenge_llm.py --carte defi --iterations 5
uv run --with nbconvert --with nbclient jupyter nbconvert --to notebook --execute --inplace comparatif_llm.ipynb
```

| Graphique (notebook) | Ce qu'il montre | Lecture |
|--------------------|-----------------|---------|
| Or cumulé / distance à l'or | Progression **intra-partie** (moyenne des runs batch) | Marches d'escalier = or ramassé ; distance qui oscille = allers-retours |
| Barres or / tours / latence | Performance **globale** par modèle (runs batch) | Part d'or 0→100 %, coût en tours et en ms |
| Courbes par itération (`defi`) | Progression **inter-parties** avec mémoire | Plateaux it. 1-2 puis montée it. 3 = coach niveau 2 débloqué |
| Trajectoires sur grille | **Mouvements** par itération (couleur = n° de partie) | Exploration erratique → chemin direct vers l'or |
| Table des raisons | **Raisonnement déclaré** vs effet réel (contrat v4) | « progresse vers l'or » + effet `bloqué` = incohérence dire/faire |

---

## `src/build_report.py` — dbt + rapport HTML

Reconstruit les couches silver/gold via dbt, puis génère `reports/benchmark.html`
(graphiques matplotlib embarqués en base64, une section par typologie).

```bash
uv run python src/build_report.py              # dbt build + rapport
uv run python src/build_report.py --skip-dbt   # rapport seul (warehouse existant)
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--skip-dbt` | ne pas relancer `dbt build` ; lit `data/warehouse.duckdb` tel quel | off |

**Prérequis :** `data/warehouse.duckdb` doit exister (créé par `dbt build` après
au moins un benchmark). Sans warehouse : lancer d'abord le benchmark puis
`build_report.py` sans `--skip-dbt`.

---

## `src/visu.py` — visualisation live (terminal)

> Parcours guidé greedy → LLM : [§ Démonstration visuelle](#démonstration-visuelle-greedy--llm).

Charge le moteur depuis le notebook et rejoue **une** partie avec le tableau de
bord `rich` (carte ASCII, historique des coups, métriques).

```bash
# Défaut : greedy, carte initiale, objectif premier or
uv run python src/visu.py

# Typologies figées
uv run python src/visu.py --carte degage --objectif tout
uv run python src/visu.py --carte ennemi_chemin --objectif tout
uv run python src/visu.py --carte leurre --objectif tout          # piège greedy (volontaire)

# Carte aléatoire
uv run python src/visu.py --carte aleatoire --objectif tout
uv run python src/visu.py --carte aleatoire --seed 7 --pause 0.3  # reproductible
uv run python src/visu.py --carte aleatoire --objectif tout --garanti
uv run python src/visu.py --carte aleatoire --taille 9 --n-or 4 --n-ennemis 2 --objectif tout --garanti

# Cerveaux LLM
uv run python src/visu.py --cerveau api --carte degage --objectif tout       # .env requis
uv run python src/visu.py --cerveau cursor --carte degage --objectif tout    # cursor-agent login
uv run python src/visu.py --cerveau llm_local --rayon 1 --carte leurre --objectif tout
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--cerveau` | `greedy` \| `cursor` \| `llm_local` \| `api` | `greedy` |
| `--rayon` | rayon de vision pour `llm_local` : `1` = fenêtre 3×3, `2` = 5×5 | `1` |
| `--model` | modèle LLM (`cursor-agent`) | `claude-4-sonnet` |
| `--carte` | `initiale`, `degage`, `ennemi_chemin`, `leurre`, `defi` | `initiale` |
| `--objectif` | `premier` (1er or) \| `tout` (tout l'or) | `premier` |
| `--max-turns` | plafond de tours | `30` |
| `--pause` | secondes entre deux tours (affichage live) | `0.5` |
| `--seed` | graine pour `--carte aleatoire` (reproductibilité) | entropie OS |
| `--taille` | côté de la grille aléatoire | `7` |
| `--n-or` | nombre d'ors (carte aléatoire) | `3` |
| `--n-ennemis` | nombre d'ennemis (carte aléatoire) | `1` |
| `--garanti` | tire des cartes aléatoires jusqu'à une gagnable par le greedy, puis rejoue en live ; affiche la seed tirée | off |

**Notes visu :**

- `--carte leurre` est **toujours la même** carte et **piege le greedy** : c'est
  le game design, pas un bug.
- `--garanti` sans `--seed` : carte **différente à chaque lancement**, toujours
  gagnable ; la seed affichée permet de rejouer la même partie avec
  `--carte aleatoire --seed <n>`.
- `--garanti --seed 42` : séquence de tirages reproductible (même carte gagnable
  trouvée à chaque fois).
- `--n-ennemis 0` : aucun piège possible ; le greedy ramasse toujours tout l'or.

En fin de partie, le script affiche : `success`, `tours`, `ors_ramasses`,
`latence_moy`, `invalides`, liste des `coups`.

---

## dbt — pipeline silver / gold

Exécuté automatiquement par `build_report.py`, ou manuellement :

```bash
uv run dbt deps --project-dir dbt --profiles-dir dbt     # packages (dbt_utils) — 1re fois
uv run dbt build --project-dir dbt --profiles-dir dbt    # seeds + modèles + tests
uv run dbt test --project-dir dbt --profiles-dir dbt       # tests seuls
uv run dbt run --project-dir dbt --profiles-dir dbt        # modèles sans tests
```

| Élément | Détail |
|---------|--------|
| Projet | `dbt/dbt_project.yml` |
| Profil | `dbt/profiles.yml` → `data/warehouse.duckdb` |
| Packages | `dbt/packages.yml` → `dbt_utils` (tests `accepted_range`, `unique_combination_of_columns`) |
| Sources bronze | parquet `data/bronze/runs/*.parquet`, `data/bronze/turns/*.parquet` |
| Seed | `modeles_llm` (référentiel modèles : éditeur, n paramètres, classe) |
| Silver | `stg_runs`, `stg_turns` (vues ; champs `iteration`, `memoire`, `raison`) |
| Gold | `kpi_run`, `kpi_typologie`, `distance_glissante` (tables) |

---

## Tests

```bash
uv run pytest                                              # moteur + pipeline de données
uv run pytest tests/test_npc_brain.py -v                   # moteur seul
uv run pytest tests/test_data_pipeline.py -v               # perception, datalog, bronze
uv run dbt test --project-dir dbt --profiles-dir dbt       # tests de qualité des données
```

Les tests pytest chargent directement les définitions du notebook
`npc_brain.ipynb` (via `src/engine.py`) : pas de copie du code, pas d'appel
réseau.

---

## Notebooks

### `npc_brain.ipynb` — moteur (source de vérité)

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace npc_brain.ipynb
```

Les cellules marquées `# [RUN]` lancent une simulation interactive. Elles sont
**ignorées** par `engine.py`, `visu.py`, `benchmark.py` et les tests, qui ne
chargent que les cellules de définition.

### `comparatif_llm.ipynb` — graphiques du challenge Ollama

Notebook **visualisation uniquement** : courbes, barres, trajectoires sur la
carte `defi`, table des raisons. Les explications (protocole coach, inputs des
IA, critères de progression, lecture des graphiques) sont dans ce README
(§ Challenge LLM) et dans [NOTES.md §16](NOTES.md#16-challenge-llm-locaux-carte-défi).

```bash
uv run python src/challenge_llm.py --carte defi --iterations 5
uv run --with nbconvert --with nbclient jupyter nbconvert --to notebook --execute --inplace comparatif_llm.ipynb
```
