# Exemple de présentation orale — npc_brain

Guide de soutenance (~12–15 min). Chaque section indique **ce qu'il faut dire**,
**ce qu'il faut montrer** et **pourquoi** (justification attendue par la spec).

> Commandes détaillées : [`README.md`](README.md) · justifications techniques :
> [`NOTES.md`](NOTES.md)

---

## Plan suggéré

| Durée | Partie | Support |
|------:|--------|---------|
| 1 min | Accroche + problématique | oral |
| 2 min | Architecture du projet | schéma arborescence (README) |
| 2 min | Game design & typologies | oral + éventuellement `--carte leurre` |
| 4 min | **Démo live greedy → LLM** | terminal `src/visu.py` |
| 2 min | Pipeline data & KPI | `reports/benchmark.html` ou schéma médaillon |
| 1 min | Tests & reproductibilité | oral |
| 1 min | Conclusion & limites | oral |

---

## 1. Accroche (30 s)

**À dire :**

> « Nous avons construit un mini-jeu sur grille : un PNJ doit ramasser de l'or
> en évitant des ennemis statiques. La brique centrale — le *cerveau* qui choisit
> le prochain coup — est interchangeable : un algorithme greedy, ou un LLM.
> L'enjeu n'est pas de faire le meilleur jeu possible, mais de **mesurer**
> objectivement le compromis entre **qualité de décision** et **coût** (latence,
> charge cognitive), via un pipeline de data engineering complet. »

**Mots-clés à placer :** grille, cerveau interchangeable, benchmark, médaillon.

---

## 2. Problématique (1 min)

**À dire :**

> « La question centrale est : *combien d'aide algorithmique faut-il donner à un
> LLM pour qu'il navigue correctement, et à quel prix ?*
>
> - Trop d'aide (perception complète + direction vers l'or) → le greedy résout
>   tout gratuitement, le LLM n'apporte rien d'intéressant.
> - Trop peu (vision locale 3×3) → le LLM doit raisonner seul, mais paye ~10 s
>   par décision.
>
> Nous avons donc **deux axes** : le **rayon de vision** (charge algo vs charge LLM)
> et la **typologie de carte** (difficulté du problème de navigation). »

---

## 3. Architecture du projet (2 min)

**À dire :**

> « Le notebook `npc_brain.ipynb` est la **source de vérité** : moteur, règles,
> cerveaux, télémétrie. Les scripts Python dans `src/` le rechargent sans
> dupliquer le code — un seul loader `engine.py`, partagé par la visu, le
> benchmark et les tests.
>
> Côté data : chaque simulation écrit du **parquet bronze** (append-only),
> dbt transforme en **silver** puis **gold** dans DuckDB, et un script génère
> un rapport HTML par typologie. »

**Schéma à dessiner ou montrer :**

```
Simulation (visu / benchmark)
        ↓  télémétrie pydantic
   Bronze (parquet runs + turns)
        ↓  dbt
   Silver (stg_runs, stg_turns)
        ↓  dbt
   Gold (kpi_run, kpi_typologie, distance_glissante)
        ↓
   reports/benchmark.html
```

**Arborescence à citer :** `npc_brain.ipynb` → `src/` → `dbt/` → `data/bronze/` → `reports/`.

---

## 4. Règles du jeu & game design (2 min)

**À dire :**

> « Nous avons **figé** le game design pour que le benchmark soit comparable :
>
> - **Plusieurs pièces d'or** (objectif `tout`), pas un seul — les parties durent
>   plus longtemps et les KPI graduent les échecs partiels.
> - **Pas de combat** — les ennemis sont des **obstacles statiques** : ils
>   bloquent un passage, point. Pas de PV, pas d'aléa de combat.
> - **Quatre typologies** de cartes, chacune teste une difficulté précise. »

| Typologie | Ce qu'elle isole | Phrase clé |
|-----------|------------------|------------|
| `degage` | trajectoire pure | « Cas favorable au greedy : aucun obstacle. » |
| `ennemi_chemin` | contournement | « L'ennemi barre l'axe direct : le greedy s'y bloque. » |
| `leurre` | résistance au piège | « L'or le plus proche (Manhattan) est emmuré ; l'or accessible est plus loin. » |
| `aleatoire` | généralisation | « Cartes seedées 9×9 : reproductibles, variées. » |

**Justification du `leurre` (si question) :**

> « Ce n'est pas un bug : c'est un piège **volontaire** pour montrer qu'un
> algorithme « toujours vers l'or le plus proche » échoue sans planification.
> La carte est identique à chaque run — c'est voulu pour le benchmark. »

---

## 5. Cerveaux & perception (1 min 30)

**À dire :**

> « Trois régimes de perception :
>
> 1. **`decide_greedy`** — perception **complète** : distances Manhattan,
>    direction vers l'or le plus proche. Décision instantanée, gratuite,
>    déterministe. C'est notre baseline.
> 2. **`decide_llm_local`** — **vision locale** : le LLM ne voit qu'une fenêtre
>    3×3 (rayon 1) ou 5×5 (rayon 2). C'est le cerveau mesuré par le benchmark.
> 3. **`decide_cursor_cli`** — LLM avec carte entière (comparaison, plus lent).
>
> Un **arbitre BFS** (`pas_optimaux`) calcule le nombre de pas optimal sur la
> carte complète — indépendamment de ce que voit le cerveau. Il sert de
> référence pour le KPI `surcout_trajectoire` : 1,0 = trajet parfait. »

**Point important pour la démo :**

> « À l'écran, le tableau de bord montre la carte **entière** à nous, humains.
> Le LLM local, lui, ne reçoit que sa fenêtre — c'est voulu pour simuler une
> perception limitée. »

---

## 6. Démo live — greedy puis LLM (4 min)

**Préparation avant la soutenance :**

```bash
uv sync
cursor-agent login    # une fois, si démo LLM prévue
```

### Partie A — Algorithme (~30 s à 1 min)

**À dire :**

> « Voici le greedy : décision algorithmique, latence nulle, aucune clé API. »

**À montrer :**

```bash
uv run python src/visu.py --carte aleatoire --objectif tout --garanti --pause 0.3
```

**Commenter pendant l'exécution :**

- Panneau **Carte** : joueur, ors, ennemis.
- Panneau **Suivi** : tour, direction, **latence ≈ 0 ms**.
- Fin : `success=True`, ors ramassés, liste des coups.

**À dire après :**

> « `--garanti` tire une carte aléatoire jusqu'à ce que le greedy puisse tout
> ramasser. La **seed affichée** me permet de rejouer exactement la même carte
> avec le LLM — comparaison à périmètre égal. »

**Noter la seed affichée** (ex. `seed tirée : 1299013684`).

**Contre-exemple rapide (optionnel, 20 s) :**

```bash
uv run python src/visu.py --carte leurre --objectif tout --pause 0.3
```

> « Même carte à chaque fois, greedy bloqué — piège game design, pas hasard. »

### Partie B — LLM (~2–4 min)

**À dire :**

> « Même carte, même objectif, mais le cerveau est un LLM à vision locale.
> Chaque décision appelle `cursor-agent` : comptez ~10 secondes par tour. »

**À montrer :**

```bash
uv run python src/visu.py --cerveau llm_local --rayon 1 \
  --carte aleatoire --seed <SEED_NOTÉE> --objectif tout --pause 1
```

**Commenter pendant l'exécution :**

- Latence **~10 000 ms** vs **0 ms** pour le greedy.
- Même carte, comportement potentiellement différent (exploration, contournement).
- `invalides=0` si le LLM respecte le contrat JSON (`PlayerDecision`).

**À dire après :**

> « On voit l'arbitrage central du projet : le greedy est gratuit mais limité ;
> le LLM peut mieux naviguer dans des cas difficiles, mais chaque décision a un
> coût mesurable. C'est exactement ce que le pipeline data agrège ensuite. »

---

## 7. Pipeline data & rapport (2 min)

**À dire :**

> « La démo live montre **une** partie. Le benchmark en lance des dizaines :
> matrice cerveau × rayon × typologie × seed. Chaque run produit deux fichiers
> parquet bronze — métadonnées du run et un enregistrement par tour.
>
> dbt nettoie (silver), calcule les KPI (gold), et exécute des tests de qualité
> (`not_null`, unicité, valeurs acceptées). Le rapport HTML se **régénère**
> entièrement à chaque exécution — pas de dashboard temps réel, conformément
> à la spec. »

**À montrer (si temps) :**

```bash
uv run python src/build_report.py --skip-dbt
# ouvrir reports/benchmark.html
```

**Points à commenter dans le rapport :**

1. **Section par typologie** — comparer `greedy` vs `llm_local_r1` vs `llm_local_r2`.
2. **Graphique arbitrage global** — axe X : latence (log), axe Y : taux d'or ramassé.
3. **Surcoût trajectoire** — greedy = 1,0 sur `degage` ; > 1 ou échec sur `leurre`.
4. **Courbe distance glissante** — convergence vers l'or ou stagnation.

**KPI à citer à l'oral (au moins 3) :**

| KPI | Une phrase |
|-----|------------|
| `taux_or_ramasse` | « Quelle fraction de l'or a été ramassée — gradue les échecs. » |
| `surcout_trajectoire` | « Ratio pas réels / pas BFS optimaux — 1,0 = parfait. » |
| `latence_moyenne_ms` | « Prix d'une décision LLM vs 0 ms pour le greedy. » |
| `deplacements_inutiles` | « Coups qui n'approchent pas de l'or — hésitations. » |

---

## 8. Tests & reproductibilité (1 min)

**À dire :**

> « 43 tests pytest chargent le moteur **directement depuis le notebook** — pas
> de copie du code, pas d'appel réseau pendant les tests. 32 tests dbt (dont
> dbt_utils) vérifient la qualité des données silver/gold. Les seeds rendent les
> cartes aléatoires reproductibles ; le schéma pydantic est versionné et
> l'évolution v1 → v2 est **réellement présente** dans le bronze committé
> (`SCHEMA_VERSION`, `union_by_name`, COALESCE silver, test pytest dédié). »

**Si on vous demande « comment je relance chez moi ? » :**

```bash
uv sync
uv run pytest
uv run python src/benchmark.py --cerveaux greedy --seeds 3
uv run python src/build_report.py
```

---

## 9. Conclusion (1 min)

**À dire :**

> « En résumé :
>
> 1. Un **moteur de jeu** modulaire (notebook source de vérité, cerveaux
>    interchangeables).
> 2. Un **game design figé** avec typologies qui isolent chaque difficulté.
> 3. Un **benchmark reproductible** avec arbitrage qualité/coût mesuré par des
>    KPI métier, pas seulement un taux de victoire.
> 4. Un **pipeline data complet** bronze → silver → gold → rapport HTML.
>
> Limites assumées : ennemis statiques, lots LLM volontairement petits
> (~10 s/décision via CLI), nombre de paramètres non publié pour les modèles
> propriétaires — ce sont des choix de périmètre, pas des oublis. »

---

## 10. Questions fréquentes (anticipation)

| Question | Réponse courte |
|----------|----------------|
| Pourquoi le greedy échoue sur `leurre` ? | Piège volontaire : l'or Manhattan-le-plus-proche est inaccessible ; il faut viser l'or lointain. |
| Pourquoi `--seed` donne toujours la même carte ? | Reproductibilité benchmark — ce n'est pas un défaut de l'aléatoire. |
| Pourquoi `--garanti` ? | Carte aléatoire **et** partie gagnable pour la démo ; affiche la seed à réutiliser. |
| Le LLM voit-il toute la carte à l'écran ? | Non en entrée (`llm_local`) ; oui pour l'observateur (dashboard). |
| Pourquoi pas de dashboard temps réel ? | Spec : rapport régénéré après `dbt build`, pas de streaming live des KPI. |
| Pourquoi parquet + DuckDB + dbt ? | Architecture médaillon : bronze immuable, transformations testées, KPI en gold. |
| Différence `benchmark.py` vs `visu.py` ? | `visu.py` = une partie live ; `benchmark.py` = matrice batch silencieuse → bronze. |
| Le KPI « nombre de paramètres du modèle » ? | Seed dbt `modeles_llm` joint au gold : `0` pour la baseline, ~1000 Mds pour kimi-k2.7 (open-weights), `NULL` documenté pour les modèles propriétaires (non publié). |
| L'axe « modèles LLM » est-il exploité ? | Oui : deux modèles benchmarkés (`sonnet-4` Anthropic, `gpt-5-mini` OpenAI) sur les mêmes typologies, comparables dans le rapport (variante `llm_local_r1_<modele>`). |
| L'évolution de schéma a-t-elle été testée ? | Oui, en réel : le bronze committé mélange v1 et v2 (`n_or_accessible` ajouté en v2, calculé par BFS) ; `union_by_name` + COALESCE silver absorbent, test pytest dédié. |
| Pourquoi warehouse/reports absents du repo ? | Régénérables (`build_report.py`) ; un rapport figé est committé dans `docs/benchmark_exemple.html` pour consultation sans rien lancer. |
| À quoi sert `packages.yml` ? | Dépendance `dbt_utils` : tests `accepted_range` (taux 0-1, latences ≥ 0) et `unique_combination_of_columns` (typologie, variante). |

---

## 11. Check-list avant la soutenance

- [ ] `uv sync` exécuté
- [ ] Démo greedy testée (`--garanti`, seed notée)
- [ ] `cursor-agent login` fait si démo LLM
- [ ] Démo LLM testée sur la même seed (vérifier le temps total < 5 min)
- [ ] `reports/benchmark.html` ouvert dans le navigateur (onglet prêt)
- [ ] Terminal en plein écran, police lisible
- [ ] `--pause 0.3` (greedy) et `--pause 1` (LLM) pour laisser le temps de commenter

---

## 12. Script condensé (5 min chrono)

Si le temps est court, enchaîner dans cet ordre :

```bash
# 1. Greedy gagnant (~15 s)
uv run python src/visu.py --carte aleatoire --objectif tout --garanti --pause 0.2

# 2. Piège greedy (~10 s, optionnel)
uv run python src/visu.py --carte leurre --objectif tout --pause 0.2

# 3. LLM même carte (~2 min, remplacer SEED)
uv run python src/visu.py --cerveau llm_local --rayon 1 \
  --carte aleatoire --seed <SEED> --objectif tout --pause 0.5

# 4. Rapport agrégé (navigateur déjà ouvert)
# reports/benchmark.html → arbitrage global + typologie leurre
```

**Fil rouge oral :** *baseline gratuite → limites du greedy → LLM plus lent mais
plus adaptable → mesure objective via le pipeline data.*
