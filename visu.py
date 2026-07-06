"""Visualisation en direct de la simulation dans le terminal.

Charge le moteur depuis `npc_brain.ipynb` (mêmes définitions que le notebook,
sans les cellules d'exécution marquées `# [RUN]`) et rejoue une partie avec le
tableau de bord live `rich`.

Exemples :
    uv run python visu.py                         # greedy, carte initiale, 1er or
    uv run python visu.py --cerveau cursor        # modèle Cursor (lent)
    uv run python visu.py --carte aleatoire --seed 7 --pause 0.3
    uv run python visu.py --objectif tout         # ramasser TOUT l'or (partie plus longue)
    uv run python visu.py --carte aleatoire --taille 11 --n-or 6 --n-ennemis 3 --objectif tout
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LLM_API_URL", "http://localhost/v1")
os.environ.setdefault("LLM_API_TOKEN", "test")

NB = Path(__file__).parent / "npc_brain.ipynb"


def load_engine() -> dict:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ns: dict = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if "# [RUN]" in src:
            continue
        if "game_loop(" in src and "def game_loop" not in src:
            continue
        exec(compile(src, "<notebook>", "exec"), ns)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description="Suivi en direct de la simulation NPC.")
    ap.add_argument("--cerveau", choices=["greedy", "cursor"], default="greedy")
    ap.add_argument("--carte", choices=["initiale", "aleatoire"], default="initiale")
    ap.add_argument("--objectif", choices=["premier", "tout"], default="premier",
                    help="'premier' = s'arrête au 1er or ; 'tout' = ramasse tout l'or")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--pause", type=float, default=0.5, help="secondes entre deux tours")
    ap.add_argument("--seed", type=int, default=None, help="graine pour --carte aleatoire")
    ap.add_argument("--taille", type=int, default=7, help="côté de la carte aléatoire")
    ap.add_argument("--n-or", type=int, default=3, help="nombre d'ors (carte aléatoire)")
    ap.add_argument("--n-ennemis", type=int, default=1, help="nombre d'ennemis (carte aléatoire)")
    args = ap.parse_args()

    ns = load_engine()

    if args.carte == "aleatoire":
        rng = ns["np"].random.default_rng(args.seed)
        world = ns["carte_aleatoire"](
            taille=args.taille, n_or=args.n_or, n_ennemis=args.n_ennemis, rng=rng
        )
    else:
        world = ns["initial_map"]

    if args.cerveau == "cursor":
        if not ns["cursor_pret"]():
            raise SystemExit("Cursor non authentifié : lance `cursor-agent login` d'abord.")
        cerveau = ns["decide_cursor_cli"]
    else:
        cerveau = ns["decide_greedy"]

    res = ns["game_loop_suivi"](
        world, decide_fn=cerveau, max_turns=args.max_turns, live=True,
        pause=args.pause, objectif=args.objectif,
    )

    print(
        f"\nRésultat : success={res['success']} | tours={res['turns']} | "
        f"ors_ramasses={res['ors_ramasses']} | latence_moy={res['latence_moyenne_ms']} ms | "
        f"invalides={res['invalides']} | coups={res['moves']}"
    )


if __name__ == "__main__":
    main()
