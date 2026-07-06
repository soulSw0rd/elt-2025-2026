# npc_brain

Simulation d'un jeu sur grille où un PNJ doit ramasser de l'or, avec un cerveau
de décision interchangeable, **benchmarkée de bout en bout** par un pipeline de
data ingénierie en architecture médaillon (parquet → DuckDB → dbt) et un
reporting par typologie de simulation.

- **`decide_greedy`** — baseline algorithmique (perception complète, instantané) ;
- **`decide_llm_local`** — LLM à **vision locale** (fenêtre autour du joueur, via `cursor-agent`) ;
- **`decide_cursor_cli`** / **`decide`** — LLM à perception complète (Cursor CLI / API OpenAI).

> **Démo visuelle (greedy puis LLM)** : voir [§ Démonstration visuelle](#démonstration-visuelle-greedy--llm).
> Justifications détaillées (game design, perception, KPI, schéma de données) :
> [`NOTES.md`](NOTES.md)

## Évaluation rapide (correcteur)

`data/warehouse.duckdb` et `reports/` sont **git-ignorés** (régénérables) : un
clone frais ne contient pas le rapport. Deux options :

- **Sans rien lancer** : un rapport d'exemple est committé →
  [`docs/benchmark_exemple.html`](docs/benchmark_exemple.html) (copie figée de
  `reports/benchmark.html`).
- **Tout régénérer** (le bronze committé suffit, aucun LLM requis) :

```bash
uv sync
uv run python src/build_report.py   # dbt deps + dbt build + reports/benchmark.html
uv run pytest                       # 43 tests moteur + data
```

## Arborescence

```
.
├── npc_brain.ipynb     # source de vérité : moteur + cerveaux + suivi
├── src/                # outils Python (chargent le moteur depuis le notebook)
│   ├── engine.py       # loader unique du notebook (partagé visu/benchmark/tests)
│   ├── visu.py         # visualisation en direct dans le terminal
│   ├── benchmark.py    # runner : matrice de simulations -> couche bronze
│   ├── datalog.py      # contrat de données (pydantic) + écriture parquet bronze
│   └── build_report.py # dbt build + rapport HTML par typologie
├── dbt/                # pipeline médaillon dbt-duckdb (seeds + silver + gold + tests)
├── data/bronze/        # parquet bruts append-only (un fichier par run, v1 + v2)
├── reports/            # benchmark.html (rapport régénérable, git-ignoré)
├── docs/               # benchmark_exemple.html : rapport figé committé (correcteur)
├── tests/              # pytest : moteur + pipeline de données
└── pyproject.toml      # dépendances (uv)
```

## Installation

```bash
# https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync                 # crée .venv + installe toutes les dépendances
cp .env.example .env    # facultatif (cerveaux OpenAI / API uniquement)
```

| Prérequis | Détail |
|-----------|--------|
| Python | ≥ 3.13 (voir `.python-version`) |
| `uv` | gestionnaire de dépendances et d'environnement virtuel |
| `greedy` | aucune clé API, aucun compte |
| `llm_local` / `cursor` | `cursor-agent login` (pas de `CURSOR_API_KEY` requise pour le CLI) |
| `decide` (OpenAI API) | variables `LLM_API_URL` et `LLM_API_TOKEN` dans `.env` |

Le cerveau `greedy` fonctionne **sans aucune clé**. Pour les cerveaux LLM via
Cursor : `cursor-agent login`.

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

### Étape 0 — Installation (une fois)

```bash
uv sync
```

Aucune clé API n'est nécessaire pour la suite jusqu'à l'étape LLM.

### Étape 1 — Algorithme greedy (démo rapide, ~10 s)

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

### Étape 2 — Prérequis LLM (une fois)

Les cerveaux `--cerveau cursor` et `--cerveau llm_local` appellent
`cursor-agent` (~10 s par décision). Authentification :

```bash
cursor-agent login
```

Sans cette étape, `visu.py` s'arrête avec : *« Cursor non authentifié »*.

### Étape 3 — LLM sur la **même carte** (démo lente, ~2–5 min)

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

### Étape 4 — Rapport agrégé (optionnel, pas live)

Le benchmark et le HTML ne remplacent **pas** la démo live ; ils agrègent des
centaines de runs pour comparer greedy vs LLM par typologie :

```bash
uv run python src/build_report.py --skip-dbt   # si warehouse déjà à jour
# → ouvrir reports/benchmark.html
```

### Récapitulatif oral

1. **Greedy** : décision algorithmique, perception complète, gratuit, instantané.
2. **LLM** : décision par modèle, latence ~10 s/coup, vision locale ou complète.
3. **Même carte** : `--seed` (aléatoire) ou typologie figée (`degage`, `leurre`…).
4. **Live vs rapport** : `visu.py` = une partie en direct ; `benchmark.html` = KPI agrégés.

Détails des options : sections [`src/visu.py`](#srcvisupy--visualisation-live-terminal)
ci-dessous et [`NOTES.md` §8](NOTES.md#8-suivi-en-direct-rendu--démo-visuelle).

## Typologies de cartes

Les commandes `--carte` (visu) et `--typologies` (benchmark) s'appuient sur
quatre layouts **figés** plus une génération aléatoire :

| Nom | Nature | Rôle |
|-----|--------|------|
| `initiale` | layout fixe (défaut visu) | carte de départ du notebook |
| `degage` | layout fixe | terrain libre, 3 ors, aucun ennemi |
| `ennemi_chemin` | layout fixe | un ennemi barre l'axe direct joueur → or |
| `leurre` | layout fixe | l'or « proche » (Manhattan) est emmuré ; piège pour le greedy |
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

Un jeu de données bronze de démonstration est committé : le pipeline est
rejouable **sans lancer de LLM**. Le rapport se régénère à chaque exécution
(pas de temps réel : `dbt build` + rebuild de la dataviz).

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
(le bronze committé contient `sonnet-4` et `gpt-5-mini`) : le modèle est loggé
en bronze, intégré à la variante (`llm_local_r1_<modele>`) et joint au seed dbt
`modeles_llm` (éditeur, nombre de paramètres, classe de taille) dans le gold.

```bash
uv run python src/benchmark.py --cerveaux llm_local --rayons 1 \
  --typologies degage,ennemi_chemin --model gpt-5-mini
```

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

# Cerveaux LLM (lents ; cursor-agent login requis)
uv run python src/visu.py --cerveau cursor --carte degage --objectif tout
uv run python src/visu.py --cerveau llm_local --rayon 1 --carte leurre --objectif tout
uv run python src/visu.py --cerveau llm_local --rayon 2 --model gpt-5-mini --carte ennemi_chemin
```

| Option | Rôle | Défaut |
|--------|------|--------|
| `--cerveau` | `greedy` \| `cursor` (LLM perception complète) \| `llm_local` (LLM vision locale) | `greedy` |
| `--rayon` | rayon de vision pour `llm_local` : `1` = fenêtre 3×3, `2` = 5×5 | `1` |
| `--model` | modèle LLM (`cursor-agent`) | `claude-4-sonnet` |
| `--carte` | `initiale` \| `aleatoire` \| `degage` \| `ennemi_chemin` \| `leurre` | `initiale` |
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
| Silver | `stg_runs`, `stg_turns` (vues) |
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

## Notebook

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace npc_brain.ipynb
```

Les cellules marquées `# [RUN]` lancent une simulation interactive. Elles sont
**ignorées** par `engine.py`, `visu.py`, `benchmark.py` et les tests, qui ne
chargent que les cellules de définition.
