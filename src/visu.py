"""Visualisation en direct de la simulation dans le terminal.

Charge le moteur depuis `npc_brain.ipynb` (mêmes définitions que le notebook,
sans les cellules d'exécution marquées `# [RUN]`) et rejoue une partie avec le
tableau de bord live `rich`.

Exemples :
    uv run python src/visu.py                     # greedy, carte initiale, 1er or
    uv run python src/visu.py --cerveau cursor    # modèle Cursor, perception complète (lent)
    uv run python src/visu.py --cerveau llm_local --rayon 1 --carte leurre --objectif tout
    uv run python src/visu.py --carte aleatoire --seed 7 --pause 0.3
    uv run python src/visu.py --objectif tout     # ramasser TOUT l'or (partie plus longue)
    uv run python src/visu.py --carte aleatoire --taille 11 --n-or 6 --n-ennemis 3 --objectif tout
    uv run python src/visu.py --carte aleatoire --objectif tout --garanti
                                                  # carte nouvelle à chaque fois, gagnable garantie
"""
import argparse
import os
from functools import partial

os.environ.setdefault("LLM_API_URL", "http://localhost/v1")
os.environ.setdefault("LLM_API_TOKEN", "test")

from engine import load_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Suivi en direct de la simulation NPC.")
    ap.add_argument("--cerveau", choices=["greedy", "cursor", "llm_local"], default="greedy",
                    help="greedy = baseline ; cursor = LLM perception complète ; "
                         "llm_local = LLM à vision locale (fenêtre --rayon)")
    ap.add_argument("--rayon", type=int, default=1,
                    help="rayon de vision du cerveau llm_local (1 = 3x3, 2 = 5x5)")
    ap.add_argument("--model", default="sonnet-4", help="modèle LLM (cursor-agent)")
    ap.add_argument("--carte", choices=["initiale", "aleatoire", "degage", "ennemi_chemin", "leurre"],
                    default="initiale")
    ap.add_argument("--objectif", choices=["premier", "tout"], default="premier",
                    help="'premier' = s'arrête au 1er or ; 'tout' = ramasse tout l'or")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--pause", type=float, default=0.5, help="secondes entre deux tours")
    ap.add_argument("--seed", type=int, default=None, help="graine pour --carte aleatoire")
    ap.add_argument("--taille", type=int, default=7, help="côté de la carte aléatoire")
    ap.add_argument("--n-or", type=int, default=3, help="nombre d'ors (carte aléatoire)")
    ap.add_argument("--n-ennemis", type=int, default=1, help="nombre d'ennemis (carte aléatoire)")
    ap.add_argument("--garanti", action="store_true",
                    help="carte aléatoire : retire des cartes jusqu'à en trouver une que le "
                         "greedy gagne (pré-simulation silencieuse), puis la rejoue en live")
    args = ap.parse_args()

    ns = load_engine()

    if args.carte == "aleatoire":
        def _monde(seed):
            rng = ns["np"].random.default_rng(seed)
            return ns["carte_aleatoire"](
                taille=args.taille, n_or=args.n_or, n_ennemis=args.n_ennemis, rng=rng
            )

        seed = args.seed
        if args.garanti:
            # Tirage de seeds candidates : reproductible si --seed est fourni,
            # sinon entropie OS -> carte différente à chaque lancement.
            tireur = ns["np"].random.default_rng(args.seed)
            for _ in range(200):
                seed = int(tireur.integers(0, 2**32))
                essai = ns["game_loop_suivi"](
                    _monde(seed), decide_fn=ns["decide_greedy"],
                    max_turns=args.max_turns, live=False, objectif=args.objectif,
                )
                if essai["success"]:
                    break
            else:
                raise SystemExit(
                    "Aucune carte gagnable trouvée en 200 tirages : "
                    "réduis --n-ennemis ou augmente --taille / --max-turns."
                )
            print(f"Carte gagnable trouvée (seed tirée : {seed}) — "
                  f"rejouable à l'identique avec : --carte aleatoire --seed {seed}")
        world = _monde(seed)
    elif args.carte in ns["LAYOUTS"]:
        world = ns["LAYOUTS"][args.carte]
    else:
        world = ns["initial_map"]

    perception_fn = None  # défaut : perception complète
    if args.cerveau in ("cursor", "llm_local"):
        if not ns["cursor_pret"]():
            raise SystemExit("Cursor non authentifié : lance `cursor-agent login` d'abord.")
        if args.cerveau == "cursor":
            cerveau = partial(ns["decide_cursor_cli"], model=args.model)
        else:
            cerveau = partial(ns["decide_llm_local"], model=args.model, verbose=False)
            perception_fn = partial(ns["perception_locale"], rayon=args.rayon)
    else:
        cerveau = ns["decide_greedy"]

    res = ns["game_loop_suivi"](
        world, decide_fn=cerveau, max_turns=args.max_turns, live=True,
        pause=args.pause, objectif=args.objectif, perception_fn=perception_fn,
    )

    print(
        f"\nRésultat : success={res['success']} | tours={res['turns']} | "
        f"ors_ramasses={res['ors_ramasses']} | latence_moy={res['latence_moyenne_ms']} ms | "
        f"invalides={res['invalides']} | coups={res['moves']}"
    )


if __name__ == "__main__":
    main()
