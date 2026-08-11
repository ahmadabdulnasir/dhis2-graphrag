"""Vector retrieval (shared by the baseline and the hybrid pipeline).

Records are chunked at facility-month granularity into natural-language
passages, embedded, and searched by cosine similarity.

Embedding backends are pluggable:
- TfidfEmbedder (default): no heavy dependencies, fully offline
- SentenceTransformerEmbedder: `uv sync --extra embeddings` (pulls PyTorch)

FAISS is used for the index when installed; otherwise sklearn NearestNeighbors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Chunk:
    chunk_id: str
    text: str
    record_ids: list


def make_chunks(df: pd.DataFrame) -> list[Chunk]:
    """One passage per facility-month, listing all indicator values."""
    chunks = []
    for (fac, lga, state, period), grp in df.groupby(
            ["facility", "lga", "state", "period"], sort=True):
        parts = [f"{row.indicator}: {row.value:,.0f}" for row in grp.itertuples()]
        text = (f"In {period}, {fac} in {lga} LGA, {state} State reported "
                + "; ".join(parts) + ".")
        chunks.append(Chunk(chunk_id=f"{fac}|{period}", text=text,
                            record_ids=list(grp.record_id)))
    return chunks


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts).toarray().astype(np.float32)

    def transform(self, texts):
        return self.vec.transform(texts).toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def transform(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class VectorIndex:
    """Cosine-similarity index: FAISS when available, sklearn otherwise."""

    def __init__(self, embeddings: np.ndarray):
        self.emb = _normalise(embeddings)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
            self.backend = "faiss"
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(metric="cosine").fit(self.emb)
            self.backend = "sklearn"

    def search(self, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _normalise(query_emb)
        if self.backend == "faiss":
            scores, idx = self.index.search(q, k)
            return scores[0], idx[0]
        dist, idx = self.index.kneighbors(q, n_neighbors=k)
        return 1.0 - dist[0], idx[0]


class VectorRetriever:
    def __init__(self, df: pd.DataFrame, embedder=None):
        self.chunks = make_chunks(df)
        self.embedder = embedder or TfidfEmbedder()
        self.index = VectorIndex(self.embedder.fit_transform([c.text for c in self.chunks]))

    def retrieve(self, question: str, k: int = 8) -> list[tuple[Chunk, float]]:
        scores, idx = self.index.search(self.embedder.transform([question]), k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores, idx) if i >= 0]
