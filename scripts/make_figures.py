"""Generate the thesis figures.

Chapter 3 (design figures described in the write-up):
    figures/fig_3_1_architecture.png      high-level system architecture
    figures/fig_3_2_er_diagram.png        ER diagram of the source data
    figures/fig_3_3_kg_schema.png         knowledge graph schema
    figures/fig_3_4_sequence.png          query workflow sequence diagram

Chapter 5 (results figures, from results/query_results.csv):
    figures/fig_5_1_accuracy.png          accuracy by category, both systems
    figures/fig_5_2_latency_context.png   latency and context size comparison

Run:  uv run python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

GREEN, AMBER, BLUE, GRAY = "#0b6e4f", "#b07d1e", "#2b5f8a", "#5c6b7a"


def _box(ax, x, y, w, h, text, fc="#e3f2ec", ec=GREEN, fs=10, style="round,pad=0.12"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))


def fig_3_1_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")

    _box(ax, 0.2, 3.6, 1.9, 1.2, "Data sources\nDHIS2 Web API\nNHW portal\nSynthetic", fc="#eef1f4", ec=GRAY, fs=9)
    _box(ax, 2.5, 3.6, 1.9, 1.2, "1. Extraction &\nValidation\n(completeness,\nconsistency)", fs=9)
    _box(ax, 4.9, 3.6, 1.9, 1.2, "2. Knowledge Graph\nConstruction\n(Neo4j)", fs=9)
    _box(ax, 7.3, 3.6, 1.9, 1.2, "3. Hybrid Retrieval\nvector search +\ngraph traversal", fs=9)
    _box(ax, 9.0, 1.9, 1.85, 1.2, "4. Grounded\nGeneration\n(LLM + provenance)", fs=9)
    _box(ax, 6.0, 0.2, 2.4, 1.1, "5. Natural-language\ninterface (FastAPI + React)", fs=9)
    # baseline parallel path
    _box(ax, 4.9, 1.9, 1.9, 1.0, "Vector-only RAG\nbaseline (no graph)", fc="#fdf3e0", ec=AMBER, fs=9)

    _arrow(ax, 2.1, 4.2, 2.5, 4.2)
    _arrow(ax, 4.4, 4.2, 4.9, 4.2)
    _arrow(ax, 6.8, 4.2, 7.3, 4.2)
    _arrow(ax, 9.2, 3.9, 9.7, 3.1)
    _arrow(ax, 9.6, 1.9, 8.4, 1.0)
    _arrow(ax, 3.45, 3.6, 4.9, 2.6, color=AMBER, ls="--")     # extraction -> baseline
    _arrow(ax, 6.8, 2.4, 9.0, 2.4, color=AMBER, ls="--")      # baseline -> generation
    ax.text(0.2, 0.35, "solid: proposed GraphRAG path      dashed: baseline path (skips the graph)",
            fontsize=9, color=GRAY)
    fig.savefig(OUT / "fig_3_1_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_3_2_er_diagram():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")

    _box(ax, 0.4, 3.2, 2.4, 1.4,
         "ORGANISATION_UNIT\n― id, name, level\n(facility, LGA,\nstate, national)",
         fc="#e8eef5", ec=BLUE, fs=9, style="square,pad=0.1")
    _box(ax, 6.2, 3.2, 2.4, 1.2, "DATA_ELEMENT /\nINDICATOR\n― id, name",
         fc="#e8eef5", ec=BLUE, fs=9, style="square,pad=0.1")
    _box(ax, 6.2, 0.5, 2.4, 1.1, "PERIOD\n― iso_month", fc="#e8eef5", ec=BLUE,
         fs=9, style="square,pad=0.1")
    _box(ax, 3.3, 1.7, 2.4, 1.3, "DATA_VALUE\n― value\n(associative entity)",
         fc="#e3f2ec", ec=GREEN, fs=9, style="square,pad=0.1")

    _arrow(ax, 3.3, 2.6, 1.8, 3.2)          # value -> org unit
    _arrow(ax, 5.7, 2.6, 6.6, 3.2)          # value -> element
    _arrow(ax, 5.7, 2.0, 6.2, 1.3)          # value -> period
    # self-referential hierarchy
    _arrow(ax, 0.9, 3.2, 0.9, 2.4, color=BLUE)
    _arrow(ax, 0.9, 2.4, 1.6, 3.2, color=BLUE)
    ax.text(0.4, 2.15, "parent_of\n(hierarchy)", fontsize=8, color=BLUE)
    ax.text(2.2, 2.75, "reported_by (N:1)", fontsize=8, color=GRAY)
    ax.text(5.75, 2.95, "measures (N:1)", fontsize=8, color=GRAY)
    ax.text(5.5, 1.45, "for_period (N:1)", fontsize=8, color=GRAY)
    fig.savefig(OUT / "fig_3_2_er_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_3_3_kg_schema():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")

    def node(x, y, text, fc="#e3f2ec", ec=GREEN, r=0.62):
        ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=1.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)

    node(1.4, 4.2, "Disease /\nIndicator")
    node(4.2, 4.2, "DataValue\n(value)", fc="#fdf3e0", ec=AMBER)
    node(7.0, 4.2, "Facility")
    node(7.0, 1.6, "LGA")
    node(4.2, 1.6, "State")
    node(1.4, 1.6, "Region")
    node(8.9, 3.0, "Period", fc="#e8eef5", ec=BLUE)

    _arrow(ax, 3.6, 4.2, 2.05, 4.2); ax.text(2.6, 4.38, "OF_INDICATOR", fontsize=8, color=GRAY)
    _arrow(ax, 4.85, 4.2, 6.35, 4.2); ax.text(5.1, 4.38, "REPORTED_AT", fontsize=8, color=GRAY)
    _arrow(ax, 4.75, 3.85, 8.35, 3.1); ax.text(6.3, 3.28, "FOR_PERIOD", fontsize=8, color=GRAY)
    _arrow(ax, 7.0, 3.55, 7.0, 2.25); ax.text(7.15, 2.9, "LOCATED_IN", fontsize=8, color=GRAY)
    _arrow(ax, 6.35, 1.6, 4.85, 1.6); ax.text(5.2, 1.78, "LOCATED_IN", fontsize=8, color=GRAY)
    _arrow(ax, 3.55, 1.6, 2.05, 1.6); ax.text(2.5, 1.78, "PART_OF", fontsize=8, color=GRAY)

    ax.text(0.3, 0.35, "Example derived triplet:  [Malaria confirmed cases] ―[OCCURRED_IN]→ [Kano State]"
                       "   (DataValue → Facility → LGA → State)",
            fontsize=9.5, color=GREEN, family="monospace")
    fig.savefig(OUT / "fig_3_3_kg_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_3_4_sequence():
    actors = ["User", "Interface\n(React)", "Retrieval\nService", "Knowledge\nGraph (Neo4j)",
              "Vector\nIndex", "Language\nModel"]
    xs = [0.8, 2.6, 4.4, 6.2, 7.8, 9.4]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 6.4); ax.axis("off")
    for x, a in zip(xs, actors):
        _box(ax, x - 0.62, 5.5, 1.24, 0.7, a, fc="#eef1f4", ec=GRAY, fs=8.5)
        ax.plot([x, x], [0.5, 5.5], color=GRAY, lw=0.8, ls=":")

    def msg(x1, x2, y, label, ls="-"):
        _arrow(ax, x1, y, x2, y, ls=ls)
        ax.text((x1 + x2) / 2, y + 0.09, label, ha="center", fontsize=8, color="#1a2733")

    msg(xs[0], xs[1], 5.0, "natural-language question")
    msg(xs[1], xs[2], 4.5, "POST /query")
    msg(xs[2], xs[2] + 0.01, 4.15, ""); ax.text(xs[2] + 0.08, 4.15, "entity linking + intent", fontsize=8, color=GRAY)
    msg(xs[2], xs[3], 3.7, "graph traversal (Cypher)")
    msg(xs[3], xs[2], 3.35, "triplets + aggregates", ls="--")
    msg(xs[2], xs[4], 2.9, "similarity search")
    msg(xs[4], xs[2], 2.55, "top-k passages", ls="--")
    msg(xs[2], xs[5], 2.1, "merged context + question")
    msg(xs[5], xs[2], 1.75, "grounded answer", ls="--")
    msg(xs[2], xs[1], 1.3, "answer + provenance", ls="--")
    msg(xs[1], xs[0], 0.85, "answer with traceable sources", ls="--")
    fig.savefig(OUT / "fig_3_4_sequence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_results():
    import pandas as pd
    res_path = Path(__file__).resolve().parents[1] / "results" / "query_results.csv"
    if not res_path.exists():
        print("results/query_results.csv not found — run run_pipeline.py first")
        return
    res = pd.read_csv(res_path)

    # 5.1 accuracy by category
    acc = res.groupby(["category", "system"])["correct"].mean().unstack() * 100
    order = ["aggregation", "comparison", "trend", "relationship", "ambiguous"]
    acc = acc.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    import numpy as np
    x = np.arange(len(acc.index))
    w = 0.36
    base_vals = acc.get("baseline")
    graph_vals = acc.get("graphrag")
    # edge-only stubs make the zero-height baseline bars visible
    ax.bar(x - w / 2, base_vals, w, label="Vector-only baseline",
           color=AMBER, edgecolor=AMBER, linewidth=1.5)
    ax.bar(x - w / 2, [2] * len(x), w, fill=False, edgecolor=AMBER,
           linewidth=1.5, linestyle=(0, (2, 2)))
    ax.bar(x + w / 2, graph_vals, w, label="GraphRAG", color=GREEN)
    for xi, v in zip(x, base_vals):
        ax.text(xi - w / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=9,
                color=AMBER, fontweight="bold")
    for xi, v in zip(x, graph_vals):
        ax.text(xi + w / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=9,
                color=GREEN, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("")
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in acc.index], rotation=0)
    ax.yaxis.grid(True, color="#e3e8ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_1_accuracy.png", dpi=200)
    plt.close(fig)

    # 5.2 latency + context size
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    lat = res.groupby("system")["latency_s"].mean()
    ctx = res.groupby("system")["context_tokens_est"].mean()
    for ax, s, title, unit in ((axes[0], lat, "Mean query latency", "seconds"),
                               (axes[1], ctx, "Mean retrieved context size", "approx. tokens")):
        ax.bar(["Baseline", "GraphRAG"],
               [s.get("baseline", 0), s.get("graphrag", 0)], color=[AMBER, GREEN], width=0.55)
        ax.set_title(title, fontsize=11); ax.set_ylabel(unit)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_2_latency_context.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_3_1_architecture()
    fig_3_2_er_diagram()
    fig_3_3_kg_schema()
    fig_3_4_sequence()
    fig_results()
    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)
