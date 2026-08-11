"""Verify the Neo4j store against the in-memory store.

Run after `docker compose up -d`:

    uv run python scripts/verify_neo4j.py

Builds the same validated synthetic dataset into both stores, then asserts
that graph traversal on Neo4j (Cypher) returns the same aggregates as the
networkx implementation, and that the full evaluation still scores 100%
through Neo4j. Prints PASS/FAIL per check.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.extraction.synthetic import generate
from src.graph.neo4j_store import Neo4jGraphStore
from src.graph.store import InMemoryGraphStore
from src.validation.quality import validate

CHECKS = [
    ("aggregate: malaria / Kano / 2024",
     dict(indicator_id="malaria_confirmed", state="Kano", period_prefix="2024")),
    ("aggregate: deliveries / Ikeja LGA / 2023",
     dict(indicator_id="facility_delivery", lga="Ikeja", period_prefix="2023")),
    ("aggregate: malaria / North region / all",
     dict(indicator_id="malaria_confirmed", region="North")),
    ("aggregate: penta3 / all / 2025",
     dict(indicator_id="penta3", period_prefix="2025")),
]


def main():
    load_dotenv()
    print("Generating + validating synthetic data...")
    clean, report = validate(generate())
    print(f"  {report.summary()}")

    mem = InMemoryGraphStore()
    mem.build(clean)

    print("Connecting to Neo4j...")
    try:
        neo = Neo4jGraphStore.from_env()
        neo.driver.verify_connectivity()
    except Exception as e:
        print(f"FAIL: cannot connect to Neo4j ({e}).\n"
              "Is the container running? Try: docker compose up -d")
        sys.exit(1)

    print("Building graph in Neo4j (a minute or two for ~22k value nodes)...")
    t0 = time.perf_counter()
    neo.build(clean)
    print(f"  built in {time.perf_counter() - t0:.0f}s")

    failures = 0
    for name, kwargs in CHECKS:
        a = mem.aggregate(**kwargs)
        b = neo.aggregate(**kwargs)
        ok = abs(a.total - b.total) < 1e-6 and a.n_values == b.n_values
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: "
              f"memory={a.total:,.0f} ({a.n_values}), neo4j={b.total:,.0f} ({b.n_values})")
        failures += not ok

    # comparison + trend parity
    m_top = max(mem.compare("tb_notified", "lga", period_prefix="2024").items(),
                key=lambda kv: kv[1].total)
    n_top = max(neo.compare("tb_notified", "lga", period_prefix="2024").items(),
                key=lambda kv: kv[1].total)
    ok = m_top[0] == n_top[0]
    print(f"  {'PASS' if ok else 'FAIL'}  compare: top LGA for TB 2024: "
          f"memory={m_top[0]}, neo4j={n_top[0]}")
    failures += not ok

    # full evaluation through Neo4j
    print("Running full evaluation through Neo4j...")
    from src.baseline.vector_rag import VectorOnlyRAG
    from src.evaluation.runner import run_comparison
    from src.generation.answerer import default_answerer
    from src.retrieval.entity_linker import EntityLinker
    from src.retrieval.graph_retriever import GraphRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.vector import VectorRetriever

    linker = EntityLinker(neo.entity_names())
    vector = VectorRetriever(clean)
    hybrid = HybridRetriever(GraphRetriever(neo), vector, linker)
    answerer = default_answerer()
    results = run_comparison(lambda q: answerer.answer(hybrid.retrieve(q)),
                             VectorOnlyRAG(vector, linker).answer,
                             clean, out_dir="results/neo4j")
    acc = results["overall"].set_index("system").loc["graphrag", "accuracy"]
    lat = results["overall"].set_index("system").loc["graphrag", "latency_s"]
    ok = acc == 1.0
    print(f"  {'PASS' if ok else 'FAIL'}  graphrag accuracy on Neo4j: {acc:.0%} "
          f"(mean latency {lat:.2f}s, target <=10s)")
    failures += not ok

    neo.close()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
