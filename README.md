# dhis2-graphrag

Implementation of the MIT thesis project: *Enhancing the Quality and
Accessibility of Public Health Data in the DHIS2 Ecosystem: A
Graph-Augmented Retrieval Approach* (Ahmad Abdulnasir Shuaib, Miva Open
University, supervisor Dr. Laud Charles).

Pipeline (Figure 3.1 of the write-up): extraction and validation → knowledge
graph → hybrid retrieval (vector + graph traversal) → grounded generation
with provenance → comparative evaluation against a vector-only RAG baseline.

## Quick start

```bash
uv sync --link-mode=copy

# full comparative evaluation on synthetic data, offline, no keys needed
uv run python run_pipeline.py

# ask a single question (GraphRAG + baseline side by side)
uv run python run_pipeline.py --ask "What was the malaria incidence in Kano State in 2024?"

# tests
uv run --with pytest pytest -q
```

Results land in `results/` (`query_results.csv`, `summary.json`, `summary.md`).

## Data sources (all three are supported)

| Source | How | Status |
|---|---|---|
| Synthetic (Track B) | `--source synthetic` (default) | Working; seed 42, 23k records, 6 states, 2020–2025 |
| DHIS2 Web API (Track A) | `cp .env.example .env`, set credentials, `--source dhis2` | Client + analytics parser done; test against `play.dhis2.org` from your machine (blocked from this sandbox) |
| Nigeria Health Watch | `--source path/to/export.csv` | CSV loader with column mapping |

## Neo4j vs in-memory graph

The retriever talks to a `GraphStore` interface. Development and offline
evaluation use the in-memory networkx store (real graph traversal, same
semantics). For the production path:

```bash
docker compose up -d          # starts Neo4j (browser at :7474)
uv run python run_pipeline.py --neo4j
```

## LLM generation

Without `OPENAI_API_KEY`, both systems use deterministic offline answerers
(reproducible; used for tests). With a key in `.env`, both systems generate
through the LLM with identical grounding instructions, so the only
experimental difference is retrieval — the fair comparison the methodology
requires.

## Layout

```
src/
├── models.py                 # canonical record schema shared by all sources
├── extraction/               # dhis2_client, nhw_loader, synthetic
├── validation/quality.py     # WHO-DQR-style completeness/consistency checks
├── graph/                    # schema, in-memory store, neo4j store
├── retrieval/                # entity_linker, vector, graph_retriever, hybrid
├── generation/answerer.py    # offline + LLM answerers, provenance on every answer
├── baseline/vector_rag.py    # vector-only comparison system
└── evaluation/               # query set (5 categories), metrics (BLEU/ROUGE-L/RAGAS hook), runner
```

## Objective coverage (Chapter 1, §1.3)

1. Extraction + validation with completeness/consistency targets — done
   (synthetic passes 97.0% / 97.7%; real-data run pending DHIS2 access)
2. Knowledge graph preserving org-unit/indicator/period relationships — done
   (Neo4j + in-memory implementations)
3. Hybrid retrieval (FAISS-ready vector search + graph traversal) — done
   (TF-IDF default; `uv sync --extra embeddings` for sentence-transformers+FAISS)
4. Natural-language interface with provenance, ≤10s latency — answers carry
   triplets + record ids; current latency ~0.02s offline. FastAPI/React UI: next phase
5. Evaluation vs vector-only baseline (RAGAS, BLEU/ROUGE, expert review) —
   harness done with 15 starter queries; expand to 30–50 and enable RAGAS
   (`--extra ragas` + API key) for the final experiments

## Notes

- `uv sync` may print an editable-install `.pth` warning on some
  filesystems; harmless. If the venv misbehaves: `rm -rf .venv && uv sync --link-mode=copy`.
- Ground truth for synthetic queries is computed with pandas, independently
  of the graph code, so the evaluation doubles as a graph-correctness check.
