"""Chargeur unique du moteur défini dans `npc_brain.ipynb`.

`visu.py`, `benchmark.py` et les tests partagent ce loader : plutôt que de
copier le code du notebook, ils exécutent ses cellules de **définition** pour
récupérer le moteur (constantes, perception, déplacement, cerveaux, boucle de
jeu). Sont ignorées :

- les cellules marquées `# [RUN]` (elles lancent une simulation, parfois un LLM) ;
- les anciens appels `game_loop(...)` nus (idem).

`npc_brain.ipynb` reste ainsi l'unique source de vérité.
"""
import json
from pathlib import Path

NB_DEFAULT = Path(__file__).resolve().parents[1] / "npc_brain.ipynb"


def load_engine(nb_path: Path | str | None = None) -> dict:
    """Exécute les cellules de définition du notebook et renvoie leur namespace."""
    nb_path = Path(nb_path) if nb_path else NB_DEFAULT
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if "# [RUN]" in src:
            continue
        if "game_loop(" in src and "def game_loop" not in src:
            continue
        # dont_inherit : ne pas propager d'éventuels `from __future__` de
        # l'appelant aux cellules (casserait les modèles pydantic du notebook).
        exec(compile(src, "<notebook>", "exec", dont_inherit=True), ns)
    return ns
