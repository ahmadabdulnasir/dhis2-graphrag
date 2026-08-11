"""End-to-end pipeline (Figure 3.1): extraction -> validation -> knowledge
graph -> hybrid retrieval -> grounded generation -> comparative evaluation.

Usage:
    uv run python run_pipeline.py                 # synthetic data, in-memory graph
    uv run python run_pipeline.py --source dhis2  # pull from DHIS2 (needs .env)
    uv run python run_pipeline.py --neo4j         # use Neo4j (docker compose up -d)
    uv run python run_pipeline.py --ask "What was the malaria incidence in Kano State in 2024?"
"""
from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv

from src.baseline.vector_rag import VectorOnlyRAG
from src.evaluation.runner import run_comparison
from src.generation.answerer import default_answerer
from src.graph.store import InMemoryGraphStore
from src.retrieval.entity_linker import EntityLinker
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import VectorRetriever
from src.validation.quality import validate


def load_data(source: str):
    if source == "synthetic":
        from src.extraction.synthetic import generate
        return generate()
    if source == "dhis2":
        from src.extraction.dhis2_client import DHIS2Client
        client = DHIS2Client()
        info = client.system_info()
        print(f"Connected to DHIS2 {info.get('version')} at {client.base_url}")
        elements = client.data_elements(["anc", "malaria", "delivery", "penta", "tb"])[:10]
        orgs = client.org_units(level=4)[:50]
        periods = [f"2024-{m:02d}" for m in range(1, 13)]
        return client.extract([e["id"] for e in elements], [o["id"] for o in orgs], periods)
    if source.endswith(".csv"):
        from src.extraction.nhw_loader import load_csv
        return load_csv(source)
    raise ValueError(f"unknown source: {source}")


def build_systems(df, use_neo4j: bool = False):
    clean, report = validate(df)
    print(f"[validation] {report.summary()}")

    if use_neo4j:
        from src.graph.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore.from_env()
    else:
        store = InMemoryGraphStore()
    t0 = time.perf_counter()
    store.build(clean)
    print(f"[graph] built in {time.perf_counter() - t0:.1f}s "
          + (str(store.stats()) if hasattr(store, "stats") else ""))

    linker = EntityLinker(store.entity_names())
    vector = VectorRetriever(clean)
    hybrid = HybridRetriever(GraphRetriever(store), vector, linker)
    answerer = default_answerer()

    graphrag = lambda q: answerer.answer(hybrid.retrieve(q))
    baseline = VectorOnlyRAG(vector, linker).answer
    return graphrag, baseline, clean, report, linker


def demo_questions(graphrag, clean):
    """Auto-generated questions over whatever entities the dataset holds,
    each verified against an independent pandas aggregate."""
    import pandas as pd  # noqa: F401

    print("\n=== Demonstration on this dataset ===")
    inds = clean[["indicator_id", "indicator"]].drop_duplicates().values[:3]
    states = sorted(clean.state.unique())
    years = sorted({p[:4] for p in clean.period})
    passed = total = 0
    for slug, name in inds:
        sub = clean[clean.indicator_id == slug]
        state = sub.state.value_counts().index[0]
        year = sorted({p[:4] for p in sub.period})[-1]
        q = f"What was the total {name} in {state} in {year}?"
        expected = sub[(sub.state == state) & sub.period.str.startswith(year)]["value"].sum()
        ans = graphrag(q)
        ok = f"{expected:,.0f}" in ans.text
        total += 1
        passed += ok
        print(f"\nQ: {q}\nA: {ans.text}")
        print(f"   [{'OK' if ok else 'MISMATCH'}] independent pandas total: {expected:,.0f}; "
              f"{ans.provenance_summary()}")
    print(f"\n{passed}/{total} spot-checks matched. "
          f"Dataset: {len(clean):,} records, {len(states)} states, years {years[0]}-{years[-1]}.")
    print("Note: the 36-query benchmark (with fixed ground truth) runs on --source synthetic.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    help="synthetic | dhis2 | path/to/nhw_export.csv")
    ap.add_argument("--neo4j", action="store_true", help="use Neo4j instead of in-memory graph")
    ap.add_argument("--ask", help="ask a single question instead of running the evaluation")
    args = ap.parse_args()

    load_dotenv()
    df = load_data(args.source)
    print(f"[extraction] {len(df):,} records from {args.source}")

    graphrag, baseline, clean, report, _ = build_systems(df, use_neo4j=args.neo4j)

    if args.ask:
        ans = graphrag(args.ask)
        print(f"\nQ: {args.ask}\nA: {ans.text}\n   ({ans.provenance_summary()})")
        b = baseline(args.ask)
        print(f"\n[baseline] {b.text}")
        return

    if args.source != "synthetic":
        # The 36-query benchmark carries ground truth tied to the synthetic
        # dataset; on other sources, run a demonstration over the actual
        # entities instead.
        demo_questions(graphrag, clean)
        return

    results = run_comparison(graphrag, baseline, clean)
    print("\n=== Overall ===")
    print(results["overall"].to_string(index=False))
    print("\n=== By category ===")
    print(results["summary"].to_string(index=False))
    print("\nSaved to results/ (query_results.csv, summary.json, summary.md)")


if __name__ == "__main__":
    main()
