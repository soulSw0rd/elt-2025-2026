"""Challenge de LLM locaux (Ollama) sur la carte par défaut.

Deux modes :

- **batch** (`--runs N`, défaut) : N parties indépendantes par modèle, sans
  mémoire — mesure la performance brute.
- **itératif** (`--iterations N`) : N parties SUCCESSIVES par modèle ; avant
  chaque partie, un résumé d'expérience est reconstruit depuis la couche
  bronze (la « database » : positions où l'or a été trouvé, cases bloquantes,
  scores passés) et injecté dans le prompt. À temperature=0, toute variation
  entre itérations vient de cette mémoire : si la courbe monte, c'est
  l'expérience qui guide. Désactivable avec `--sans-memoire` (groupe témoin).

Les résultats partent en couche bronze (cerveau="ollama", modele=<nom>,
iteration=n, memoire=true/false) : le notebook `comparatif_llm.ipynb` en tire
courbes de progression, barres comparatives et graphique d'itérations.

Prérequis :
    ollama serve                       # serveur local (localhost:11434)
    ollama pull qwen2.5:0.5b           # petits modèles de l'expérience
    ollama pull llama3.2:1b
    ollama pull gemma2:2b

Exemples :
    uv run python src/challenge_llm.py                          # batch 3x3
    uv run python src/challenge_llm.py --iterations 5           # séquence avec mémoire
    uv run python src/challenge_llm.py --iterations 5 --sans-memoire
    uv run python src/challenge_llm.py --models qwen2.5:0.5b --runs 1
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("LLM_API_URL", "http://localhost/v1")
os.environ.setdefault("LLM_API_TOKEN", "test")

from datalog import BRONZE_DIR, ecrire_bronze, nouveau_run_id, records_depuis_resultat
from engine import load_engine

MODELES_DEFAUT = "qwen2.5:0.5b,llama3.2:1b,gemma2:2b"


def carte_pour(nom: str, ns):
    if nom == "initiale":
        return ns["initial_map"]
    return ns["LAYOUTS"][nom]


def ors_accessibles(carte, ns) -> int:
    joueur = tuple(int(x) for x in ns["localize"](carte, ns["PLAYER"])[0])
    dist = ns["_bfs_distances"](carte, joueur)
    return int(sum(1 for (r, c) in ns["localize"](carte, ns["GOLD"]) if dist[r, c] >= 0))


def prochain_coup_coach(carte, ns, position) -> str | None:
    """Prochain coup optimal depuis `position` vers l'or (BFS de l'arbitre).

    Utilisé par le coach de niveau >= 2 : à chaque tour, l'arbitre indique la
    direction du plus court chemin depuis la position ACTUELLE. Le modèle n'a
    plus qu'à suivre l'instruction — on mesure alors sa capacité à exploiter
    une aide explicite (les nano-modèles ne savent pas suivre un plan
    multi-coups statique, éprouvé en itérations préliminaires).
    """
    or_pos = tuple(int(x) for x in ns["localize"](carte, ns["GOLD"])[0])
    dist = ns["_bfs_distances"](carte, or_pos)
    r, c = position
    candidats = []
    for direction, (dr, dc) in ns["MOVES"].items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < carte.shape[0] and 0 <= nc < carte.shape[1] and dist[nr, nc] >= 0:
            candidats.append((int(dist[nr, nc]), direction))
    if not candidats:
        return None
    return min(candidats)[1]


def resume_experience(dossier: Path, modele: str, typologie: str, carte, ns,
                      max_parties: int = 5) -> tuple[str | None, int, dict]:
    """Reconstruit la mémoire d'un modèle depuis la couche bronze — « coach »
    à aide graduée : chaque niveau n'est débloqué que par les parties
    précédentes, tout est relu depuis la database (aucun état RAM).

    Renvoie (texte_memoire, niveau, chemin_gagnant) :

    - niveau 0 (aucune partie antérieure) : (None, 0, {}) ;
    - niveau 1 (>= 1 échec) : cases déjà visitées + consigne anti-boucle ;
    - niveau 2 (>= 2 échecs sans or) : coach de l'arbitre — à CHAQUE tour, le
      prochain coup optimal (BFS) est placé en consigne finale du prompt
      (escalade qui rend la progression possible ; un plan statique
      multi-coups a été essayé : les nano-modèles ne le suivent pas) ;
    - niveau 3 (>= 1 partie avec or ramassé) : rejouer SA partie gagnante —
      `chemin_gagnant` mappe position -> direction jouée lors du succès, la
      consigne par tour vient de l'expérience du modèle (plus de l'arbitre).

    Seules les parties de séquences itératives comptent (`iteration` non NULL) :
    les runs batch restent un groupe témoin non pollué.
    """
    import duckdb

    runs_glob = str(dossier / "runs" / "*.parquet")
    turns_glob = str(dossier / "turns" / "*.parquet")
    if not list((dossier / "runs").glob("*.parquet")):
        return None, 0, {}

    con = duckdb.connect()
    runs_df = con.execute(f"""
        select * from read_parquet('{runs_glob}', union_by_name=true)
        where cerveau = 'ollama' and modele = ? and typologie = ?
        order by horodatage
    """, [modele, typologie]).df()
    if "iteration" not in runs_df.columns:
        return None, 0, {}
    runs_df = runs_df[runs_df["iteration"].notna()]
    if runs_df.empty:
        return None, 0, {}

    anciens = runs_df[["run_id", "ors_ramasses", "n_or_initial", "turns"]].values.tolist()
    run_ids = [r[0] for r in anciens]
    lignes_turns = con.execute(f"""
        select run_id, tour, direction, or_ramasse, bouge,
               pos_avant_row, pos_avant_col, pos_apres_row, pos_apres_col
        from read_parquet('{turns_glob}', union_by_name=true)
        where run_id in ({','.join('?' * len(run_ids))})
        order by run_id, tour
    """, run_ids).fetchall()

    visitees = set()
    chemins = {}   # run_id -> [(tour, direction, or_ramasse, pos_avant)]
    for (rid, tour, direction, or_ramasse, bouge, br, bc, ar, ac) in lignes_turns:
        visitees.add((int(ar), int(ac)))
        chemins.setdefault(rid, []).append(
            (tour, direction, or_ramasse, (int(br), int(bc)))
        )

    n_echecs_sans_or = sum(1 for (_rid, ors, _n, _t) in anciens if ors == 0)
    reussites = [rid for (rid, ors, _n, _t) in anciens if ors > 0]

    lignes = ["Bilan de tes parties précédentes :"]
    for no, (_rid, ors, n_or, turns) in enumerate(anciens[-max_parties:], 1):
        lignes.append(f"- partie {no} : {ors}/{n_or} or ramassé en {turns} tours")

    niveau, chemin_gagnant = 1, {}

    if reussites:
        # Niveau 3 : rejouer SA partie gagnante. On mémorise, pour chaque
        # position traversée lors du succès, la direction alors jouée : la
        # consigne par tour viendra de l'expérience du modèle lui-même.
        niveau = 3
        meilleur = min(
            reussites,
            key=lambda rid: next(t for (t, _d, o, _p) in chemins[rid] if o),
        )
        tour_or = next(t for (t, _d, o, _p) in chemins[meilleur] if o)
        seq = []
        for (t, d, _o, pos) in chemins[meilleur]:
            if d is not None and t <= tour_or:
                # On garde le DERNIER coup joué depuis chaque position : la
                # dernière sortie d'une case dans une trajectoire gagnante
                # mène au but (garder le premier rejouerait les détours en boucle).
                chemin_gagnant[pos] = d
                seq.append(d)
        lignes.append(
            "- Tu as DÉJÀ GAGNÉ sur cette grille : ton chemin gagnant était "
            f"{' -> '.join(seq)}. À chaque tour, rejoue le coup de ta victoire."
        )
    elif n_echecs_sans_or >= 2:
        # Niveau 2 : coach de l'arbitre par tour (consigne finale du prompt).
        niveau = 2
        lignes.append(
            "- Un COACH t'accompagne : à chaque tour, il t'indique le meilleur "
            "coup. Joue exactement la direction qu'il conseille."
        )
    else:
        # Niveau 1 : cases déjà explorées (positions uniquement, pas de mots
        # de direction) + consigne anti-boucle.
        if visitees:
            pos = ", ".join(f"({r},{c})" for r, c in sorted(visitees))
            lignes.append(f"- cases déjà visitées sans trouver d'or : {pos} — explore ailleurs")
        lignes.append("- si un coup ne fait pas bouger J, ne le rejoue pas : change de direction")

    return "\n".join(lignes), niveau, chemin_gagnant


def main() -> None:
    ap = argparse.ArgumentParser(description="Challenge de LLM locaux (Ollama) -> couche bronze.")
    ap.add_argument("--models", default=MODELES_DEFAUT,
                    help=f"modèles Ollama séparés par des virgules (défaut : {MODELES_DEFAUT})")
    ap.add_argument("--carte", choices=["initiale", "degage", "ennemi_chemin", "leurre", "defi"],
                    default="initiale",
                    help="carte du challenge ; 'defi' = 1 or, 2 ennemis (progression itérative)")
    ap.add_argument("--runs", type=int, default=3,
                    help="mode batch : répétitions indépendantes par modèle (sans mémoire)")
    ap.add_argument("--iterations", type=int, default=None,
                    help="mode itératif : parties successives par modèle, avec résumé "
                         "d'expérience relu depuis bronze avant chaque partie")
    ap.add_argument("--sans-memoire", action="store_true",
                    help="mode itératif sans injection d'expérience (groupe témoin)")
    ap.add_argument("--rayon", type=int, default=1, help="rayon de vision locale (1 = 3x3)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--objectif", choices=["premier", "tout"], default="tout")
    ap.add_argument("--sortie", type=Path, default=None,
                    help="dossier bronze (défaut : data/bronze)")
    args = ap.parse_args()

    modeles = [m.strip() for m in args.models.split(",") if m.strip()]
    mode_iteratif = args.iterations is not None
    n_parties = args.iterations if mode_iteratif else args.runs
    avec_memoire = mode_iteratif and not args.sans_memoire
    dossier_bronze = args.sortie or BRONZE_DIR

    ns = load_engine()
    if not ns["ollama_pret"]():
        sys.exit("Ollama ne répond pas sur localhost:11434 : lance `ollama serve` "
                 "puis `ollama pull <modele>` pour chaque modèle du challenge.")

    carte = carte_pour(args.carte, ns)
    perception_fn = partial(ns["perception_locale"], rayon=args.rayon)
    total = len(modeles) * n_parties

    mode = (f"itératif ({'avec' if avec_memoire else 'sans'} mémoire)"
            if mode_iteratif else "batch (sans mémoire)")
    print(f"Challenge {mode} : {len(modeles)} modèle(s) x {n_parties} partie(s) "
          f"sur carte '{args.carte}' (objectif={args.objectif}, rayon={args.rayon})")

    i = 0
    for modele in modeles:
        for partie_no in range(1, n_parties + 1):
            i += 1
            experience, niveau, chemin_gagnant = None, 0, {}
            if avec_memoire:
                experience, niveau, chemin_gagnant = resume_experience(
                    dossier_bronze, modele, args.carte, carte, ns)
            etiquette = f"iter {partie_no}" if mode_iteratif else f"run {partie_no}"
            memo = f" [mémoire n{niveau}]" if experience else ""
            print(f"[{i}/{total}] {modele} · {etiquette}{memo} ... ", end="", flush=True)

            if niveau == 2:
                # Coach de l'arbitre : prochain coup optimal (BFS) recalculé à
                # CHAQUE tour, placé en consigne FINALE du prompt (position à
                # plus fort poids pour les petits modèles).
                def decide_fn(p, _modele=modele, _exp=experience):
                    pos = (p["position"]["row"], p["position"]["col"])
                    coup = prochain_coup_coach(carte, ns, pos)
                    consigne = (f"CONSIGNE DU COACH, prioritaire sur tout le reste : "
                                f"joue {coup} à ce tour.") if coup else None
                    return ns["decide_ollama"](p, model=_modele, experience=_exp,
                                               consigne_finale=consigne)
            elif niveau == 3:
                # Rejouer sa victoire : à chaque position déjà traversée lors
                # de la partie gagnante, on rappelle le coup alors joué. Si le
                # modèle dévie du chemin (position inconnue), le coach BFS de
                # l'arbitre le ramène vers l'or.
                def decide_fn(p, _modele=modele, _exp=experience, _cg=chemin_gagnant):
                    pos = (p["position"]["row"], p["position"]["col"])
                    coup = _cg.get(pos)
                    if coup:
                        consigne = ("CONSIGNE, prioritaire sur tout le reste : depuis "
                                    "cette position, ta partie GAGNANTE avait joué "
                                    f"{coup} — joue {coup} à ce tour.")
                    else:
                        secours = prochain_coup_coach(carte, ns, pos)
                        consigne = ("CONSIGNE DU COACH, prioritaire sur tout le reste : "
                                    f"joue {secours} à ce tour.") if secours else None
                    return ns["decide_ollama"](p, model=_modele, experience=_exp,
                                               consigne_finale=consigne)
            else:
                decide_fn = partial(ns["decide_ollama"], model=modele, experience=experience)
            resultat = ns["game_loop_suivi"](
                carte, decide_fn=decide_fn, perception_fn=perception_fn,
                max_turns=args.max_turns, live=False, objectif=args.objectif,
            )
            run, turns = records_depuis_resultat(
                resultat,
                run_id=nouveau_run_id(),
                cerveau="ollama",
                modele=modele,
                rayon_vision=args.rayon,
                typologie=args.carte,
                seed=None,
                taille_carte=int(carte.shape[0]),
                n_or_initial=int((carte == ns["GOLD"]).sum()),
                n_ennemis=int((carte == ns["ENNEMY"]).sum()),
                max_turns=args.max_turns,
                objectif=args.objectif,
                pas_optimaux=ns["pas_optimaux"](carte, objectif=args.objectif),
                n_or_accessible=ors_accessibles(carte, ns),
                iteration=partie_no if mode_iteratif else None,
                memoire=avec_memoire if mode_iteratif else None,
            )
            ecrire_bronze(run, turns, dossier=args.sortie)
            print(f"success={run.success} ors={run.ors_ramasses}/{run.n_or_initial} "
                  f"turns={run.turns} latence={run.latence_moyenne_ms}ms "
                  f"invalides={run.invalides}")

    print(f"\nBronze écrit dans : {dossier_bronze}")
    print("Analyse : ouvrir comparatif_llm.ipynb (courbes + barres + itérations).")


if __name__ == "__main__":
    main()
