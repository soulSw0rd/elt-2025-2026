"""Runner du benchmark : exécute la matrice de simulations et écrit la couche bronze.

Matrice = cerveau × rayon de vision × typologie de carte × seeds.

- `greedy`    : baseline algorithmique, perception COMPLÈTE (delta directionnel),
  gratuite -> on peut multiplier les seeds.
- `llm_local` : cerveau LLM à vision LOCALE (fenêtre rayon 1 ou 2), ~10 s par
  décision via `cursor-agent` -> peu de seeds, c'est voulu (spec : point
  d'équilibre, pas qualité maximale).

Exemples :
    uv run python src/benchmark.py --dry-run                  # affiche la matrice
    uv run python src/benchmark.py                            # greedy seul (rapide)
    uv run python src/benchmark.py --cerveaux greedy,llm_local --rayons 1,2 --seeds 2
    uv run python src/benchmark.py --typologies leurre --cerveaux llm_local --rayons 2
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("LLM_API_URL", "http://localhost/v1")
os.environ.setdefault("LLM_API_TOKEN", "test")

from datalog import ecrire_bronze, nouveau_run_id, records_depuis_resultat
from engine import load_engine

TYPOLOGIES = ["degage", "ennemi_chemin", "leurre", "aleatoire"]
TAILLE_ALEATOIRE = 9
N_OR_ALEATOIRE = 3
N_ENNEMIS_ALEATOIRE = 2


def construire_matrice(args, ns) -> list[dict]:
    """Développe la matrice cerveau × rayon × typologie × seed en configs de run."""
    configs = []
    for typologie in args.typologies:
        # Les layouts fixes n'ont qu'une instance ; l'aléatoire varie par seed.
        seeds = list(range(args.seeds)) if typologie == "aleatoire" else [None]
        for cerveau in args.cerveaux:
            # Le rayon de vision n'a de sens que pour le cerveau à vision locale.
            rayons = args.rayons if cerveau == "llm_local" else [None]
            for rayon in rayons:
                for seed in seeds:
                    configs.append({
                        "cerveau": cerveau,
                        "rayon_vision": rayon,
                        "typologie": typologie,
                        "seed": seed,
                        "modele": args.model if cerveau == "llm_local" else None,
                    })
    return configs


def carte_pour(config, ns):
    if config["typologie"] == "aleatoire":
        rng = ns["np"].random.default_rng(config["seed"])
        return ns["carte_aleatoire"](
            taille=TAILLE_ALEATOIRE, n_or=N_OR_ALEATOIRE,
            n_ennemis=N_ENNEMIS_ALEATOIRE, rng=rng,
        )
    return ns["LAYOUTS"][config["typologie"]]


def executer_run(config, args, ns) -> dict:
    carte = carte_pour(config, ns)

    if config["cerveau"] == "llm_local":
        decide_fn = partial(ns["decide_llm_local"], model=config["modele"], verbose=False)
        perception_fn = partial(ns["perception_locale"], rayon=config["rayon_vision"])
    elif config["cerveau"] == "greedy":
        decide_fn = ns["decide_greedy"]
        perception_fn = None  # perception complète (baseline)
    else:
        raise ValueError(f"cerveau inconnu : {config['cerveau']}")

    resultat = ns["game_loop_suivi"](
        carte, decide_fn=decide_fn, perception_fn=perception_fn,
        max_turns=args.max_turns, live=False, objectif=args.objectif,
    )

    run, turns = records_depuis_resultat(
        resultat,
        run_id=nouveau_run_id(),
        cerveau=config["cerveau"],
        modele=config["modele"],
        rayon_vision=config["rayon_vision"],
        typologie=config["typologie"],
        seed=config["seed"],
        taille_carte=int(carte.shape[0]),
        n_or_initial=int((carte == ns["GOLD"]).sum()),
        n_ennemis=int((carte == ns["ENNEMY"]).sum()),
        max_turns=args.max_turns,
        objectif=args.objectif,
        pas_optimaux=ns["pas_optimaux"](carte, objectif=args.objectif),
    )
    chemins = ecrire_bronze(run, turns, dossier=args.sortie)
    return {"run": run, "chemins": chemins}


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark npc_brain -> couche bronze (parquet).")
    ap.add_argument("--cerveaux", default="greedy",
                    help="liste séparée par des virgules : greedy,llm_local")
    ap.add_argument("--rayons", default="1,2",
                    help="rayons de vision du cerveau llm_local (ex. 1,2)")
    ap.add_argument("--typologies", default=",".join(TYPOLOGIES),
                    help=f"liste parmi : {','.join(TYPOLOGIES)}")
    ap.add_argument("--seeds", type=int, default=5,
                    help="nombre de seeds pour la typologie aleatoire")
    ap.add_argument("--model", default="sonnet-4", help="modèle LLM (cursor-agent)")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--objectif", choices=["premier", "tout"], default="tout")
    ap.add_argument("--sortie", type=Path, default=None,
                    help="dossier bronze (défaut : data/bronze)")
    ap.add_argument("--dry-run", action="store_true",
                    help="affiche la matrice de runs sans exécuter")
    args = ap.parse_args()

    args.cerveaux = [c.strip() for c in args.cerveaux.split(",") if c.strip()]
    args.rayons = [int(r) for r in args.rayons.split(",") if r.strip()]
    args.typologies = [t.strip() for t in args.typologies.split(",") if t.strip()]

    inconnues = set(args.typologies) - set(TYPOLOGIES)
    if inconnues:
        sys.exit(f"typologies inconnues : {sorted(inconnues)}")

    ns = load_engine()
    configs = construire_matrice(args, ns)

    print(f"Matrice : {len(configs)} run(s) "
          f"(cerveaux={args.cerveaux}, rayons={args.rayons}, "
          f"typologies={args.typologies}, seeds={args.seeds}, objectif={args.objectif})")

    if args.dry_run:
        for i, c in enumerate(configs, 1):
            print(f"  {i:3d}. {c}")
        return

    if "llm_local" in args.cerveaux and not ns["cursor_pret"]():
        sys.exit("Cursor non authentifié (`cursor-agent login`) : "
                 "impossible de lancer le cerveau llm_local.")

    for i, config in enumerate(configs, 1):
        etiquette = (f"{config['cerveau']}"
                     + (f"/r{config['rayon_vision']}" if config["rayon_vision"] else "")
                     + f" · {config['typologie']}"
                     + (f" · seed={config['seed']}" if config["seed"] is not None else ""))
        print(f"[{i}/{len(configs)}] {etiquette} ... ", end="", flush=True)
        sortie = executer_run(config, args, ns)
        r = sortie["run"]
        print(f"success={r.success} ors={r.ors_ramasses}/{r.n_or_initial} "
              f"turns={r.turns} latence={r.latence_moyenne_ms}ms")

    print(f"\nBronze écrit dans : {args.sortie or 'data/bronze'}")


if __name__ == "__main__":
    main()
