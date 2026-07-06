"""Reporting du benchmark : dbt build (silver + gold) puis rapport HTML autonome.

Une section par TYPOLOGIE de simulation (degage, ennemi_chemin, leurre,
aleatoire), comparant les variantes de cerveau (greedy, llm_local_r1, ...).

Le rapport se régénère à chaque exécution — après chaque nouveau lot de runs
ou changement de paramètres, relancer simplement :

    uv run python src/build_report.py            # dbt build + rapport
    uv run python src/build_report.py --skip-dbt # rapport seul

Sortie : reports/benchmark.html (images matplotlib embarquées en base64).
"""
import argparse
import base64
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[1]
WAREHOUSE = RACINE / "data" / "warehouse.duckdb"
SORTIE = RACINE / "reports" / "benchmark.html"

ORDRE_TYPOLOGIES = ["degage", "ennemi_chemin", "leurre", "aleatoire"]

DESCRIPTIONS = {
    "degage": "Terrain libre, 3 ors, aucun ennemi : mesure la trajectoire pure (cas favorable à la baseline).",
    "ennemi_chemin": "Un ennemi barre l'axe direct joueur → or : mesure la capacité de contournement.",
    "leurre": "L'or « proche » (Manhattan) est emmuré ; l'or accessible est plus loin : mesure la résistance au piège.",
    "aleatoire": "Cartes 9×9 seedées (3 ors, 2 ennemis) : mesure la généralisation.",
}


def lancer_dbt() -> None:
    """Reconstruit silver + gold depuis le bronze (et exécute les tests dbt)."""
    print(">> dbt build (silver + gold)")
    res = subprocess.run(
        ["dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"],
        cwd=RACINE,
    )
    if res.returncode != 0:
        sys.exit("dbt build a échoué : rapport non généré (données non fiables).")


def fig_en_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image(fig) -> str:
    return f'<img src="data:image/png;base64,{fig_en_base64(fig)}" />'


def barres(df, colonnes_titres, titre) -> str:
    """Bar charts côte à côte : une barre par variante, un panneau par KPI."""
    variantes = df["variante"].tolist()
    n = len(colonnes_titres)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (col, sous_titre) in zip(axes, colonnes_titres):
        valeurs = df[col].fillna(0.0)
        b = ax.bar(variantes, valeurs, color=plt.cm.tab10.colors[: len(variantes)])
        ax.set_title(sous_titre, fontsize=10)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    fig.suptitle(titre, fontsize=12)
    fig.tight_layout()
    return image(fig)


def courbe_distance(df_dist, typologie) -> str:
    """Moyenne glissante de la distance à l'or : une courbe par variante."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    sous = df_dist[df_dist["typologie"] == typologie]
    for i, (variante, grp) in enumerate(sous.groupby("variante")):
        moy = grp.groupby("tour")["distance_glissante_3"].mean()
        ax.plot(moy.index, moy.values, marker="o", markersize=3,
                label=variante, color=plt.cm.tab10.colors[i % 10])
    ax.set_xlabel("tour")
    ax.set_ylabel("distance à l'or (glissante 3)")
    ax.set_title("Convergence vers l'or (moyenne glissante sur 3 tours)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return image(fig)


def tableau_html(df) -> str:
    return df.to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0)


def arbitrage_global(df_typo) -> str:
    """Le graphe central du projet : qualité (taux d'or ramassé) vs coût (latence)."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    agg = df_typo.groupby("variante").agg(
        latence=("latence_moyenne_ms", "mean"),
        qualite=("taux_or_ramasse", "mean"),
    ).reset_index()
    for i, ligne in agg.iterrows():
        x = max(float(ligne["latence"]), 0.5)  # échelle log : 0 ms -> 0.5 ms
        ax.scatter(x, ligne["qualite"], s=140, color=plt.cm.tab10.colors[i % 10],
                   zorder=3, label=ligne["variante"])
        ax.annotate(ligne["variante"], (x, ligne["qualite"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("latence moyenne par décision (ms, échelle log)")
    ax.set_ylabel("taux d'or ramassé (moyenne toutes typologies)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Arbitrage central : qualité de décision vs coût de décision")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return image(fig)


GABARIT = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Benchmark npc_brain</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1080px;
         color: #1a1a1a; line-height: 1.5; }}
  h1 {{ border-bottom: 3px solid #4f81bd; padding-bottom: .3rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }}
  .meta {{ color: #666; font-size: .9rem; }}
  .desc {{ background: #f4f7fb; border-left: 4px solid #4f81bd; padding: .6rem 1rem;
          margin: .8rem 0; }}
  img {{ max-width: 100%; display: block; margin: 1rem 0; }}
  table {{ border-collapse: collapse; font-size: .85rem; margin: 1rem 0; }}
  th, td {{ padding: .35rem .7rem; text-align: right; }}
  th {{ background: #4f81bd; color: white; }}
  tr:nth-child(even) {{ background: #f0f4fa; }}
</style>
</head>
<body>
<h1>Benchmark npc_brain — rapport par typologie</h1>
<p class="meta">Généré le {date} · {n_runs} runs en base · pipeline parquet → DuckDB → dbt (bronze/silver/gold)</p>
<p>Chaque section compare les <strong>variantes de cerveau</strong> (baseline
algorithmique <code>greedy</code> à perception complète, LLM à vision locale
<code>llm_local_rN</code> où N est le rayon de vision) sur une typologie de carte.
Le KPI de trajectoire se lit contre l'arbitre BFS : surcoût 1.0 = trajet parfait.</p>

<h2>Arbitrage global qualité / coût</h2>
{arbitrage}

{sections}

<h2>Données gold complètes (kpi_typologie)</h2>
{tableau_global}
</body>
</html>
"""

SECTION = """
<h2>Typologie « {typologie} »</h2>
<div class="desc">{description} <em>({n_runs} runs)</em></div>
{barres_qualite}
{barres_efficacite}
{courbe}
"""


def construire_rapport() -> None:
    if not WAREHOUSE.exists():
        sys.exit(f"{WAREHOUSE} introuvable : lancer d'abord le benchmark puis dbt build.")

    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df_typo = con.sql("select * from main_gold.kpi_typologie").df()
    df_dist = con.sql("select * from main_gold.distance_glissante").df()
    n_runs = con.sql("select count(*) from main_gold.kpi_run").fetchone()[0]
    con.close()

    sections = []
    for typologie in ORDRE_TYPOLOGIES:
        df = df_typo[df_typo["typologie"] == typologie].sort_values("variante")
        if df.empty:
            continue
        sections.append(SECTION.format(
            typologie=typologie,
            description=DESCRIPTIONS.get(typologie, ""),
            n_runs=int(df["n_runs"].sum()),
            barres_qualite=barres(df, [
                ("taux_reussite", "Taux de réussite (tout l'or)"),
                ("taux_or_ramasse", "Taux d'or ramassé"),
                ("latence_moyenne_ms", "Latence moyenne / décision (ms)"),
            ], "Qualité et coût de décision"),
            barres_efficacite=barres(df, [
                ("pas_par_piece_moyen", "Pas par pièce ramassée"),
                ("surcout_trajectoire_moyen", "Surcoût vs BFS optimal"),
                ("deplacements_inutiles_moyens", "Déplacements inutiles / run"),
            ], "Efficacité de trajectoire"),
            courbe=courbe_distance(df_dist, typologie),
        ))

    colonnes = ["typologie", "variante", "modele", "n_runs", "taux_reussite",
                "taux_or_ramasse", "pas_par_piece_moyen", "surcout_trajectoire_moyen",
                "deplacements_inutiles_moyens", "coups_bloques_moyens",
                "invalides_moyens", "latence_moyenne_ms", "latence_p95_ms"]

    html = GABARIT.format(
        date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        n_runs=n_runs,
        arbitrage=arbitrage_global(df_typo),
        sections="\n".join(sections),
        tableau_global=tableau_html(
            df_typo.sort_values(["typologie", "variante"])[colonnes]
        ),
    )

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(html, encoding="utf-8")
    print(f">> rapport écrit : {SORTIE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="dbt build + rapport HTML du benchmark.")
    ap.add_argument("--skip-dbt", action="store_true",
                    help="ne pas relancer dbt (réutilise le warehouse existant)")
    args = ap.parse_args()

    if not args.skip_dbt:
        lancer_dbt()
    construire_rapport()


if __name__ == "__main__":
    main()
