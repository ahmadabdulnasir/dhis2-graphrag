"""FastAPI backend — the natural-language interface (objective 4, stage 5).

Endpoints:
    GET  /health            liveness + graph stats + validation report
    GET  /entities          known states, LGAs, regions, indicators
    POST /query             GraphRAG answer with triplets + provenance + latency
    POST /baseline/query    vector-only baseline (for side-by-side demos)

Run:
    uv run uvicorn src.api.main:app --reload
Environment:
    GRAPHRAG_SOURCE  synthetic (default) | dhis2 | path/to/export.csv
    GRAPHRAG_NEO4J   set to 1 to use Neo4j instead of the in-memory graph
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MAX_PROVENANCE = 25


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: str
    triplets: list[str]
    record_ids: list[str]
    n_supporting_values: int
    mode: str
    latency_s: float


class State:
    graphrag = None
    baseline = None
    linker = None
    report = None
    stats = None


state = State()


def _build():
    load_dotenv()
    from run_pipeline import build_systems, load_data

    source = os.getenv("GRAPHRAG_SOURCE", "synthetic")
    use_neo4j = os.getenv("GRAPHRAG_NEO4J", "") in ("1", "true", "yes")
    df = load_data(source)
    graphrag, baseline, clean, report, linker = build_systems(df, use_neo4j=use_neo4j)
    state.graphrag, state.baseline = graphrag, baseline
    state.report, state.linker = report, linker
    state.stats = {"source": source, "neo4j": use_neo4j,
                   "n_clean_records": int(len(clean))}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _build()
    yield


app = FastAPI(title="DHIS2 GraphRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


def _answer(fn, question: str) -> QueryResponse:
    if fn is None:
        raise HTTPException(503, "pipeline not initialised")
    t0 = time.perf_counter()
    try:
        ans = fn(question)
    except Exception as e:
        raise HTTPException(500, f"query failed: {e}") from e
    latency = time.perf_counter() - t0
    intent = state.linker.link(question).intent if state.linker else "unknown"
    return QueryResponse(
        question=question, answer=ans.text,
        intent=intent,
        triplets=ans.triplets[:MAX_PROVENANCE],
        record_ids=ans.record_ids[:MAX_PROVENANCE],
        n_supporting_values=ans.n_supporting_values,
        mode=ans.mode, latency_s=round(latency, 4),
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if state.graphrag else "initialising",
        "validation": state.report.summary() if state.report else None,
        "validation_targets_met": state.report.meets_targets if state.report else None,
        **(state.stats or {}),
    }


@app.get("/entities")
def entities():
    if state.linker is None:
        raise HTTPException(503, "pipeline not initialised")
    return {"states": state.linker.states, "lgas": state.linker.lgas,
            "indicators": state.linker.indicators}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return _answer(state.graphrag, req.question)


@app.post("/baseline/query", response_model=QueryResponse)
def baseline_query(req: QueryRequest):
    return _answer(state.baseline, req.question)


# ---- frontend (single-file React app, no build step) ----
_STATIC = Path(__file__).resolve().parents[2] / "static"
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="frontend")
