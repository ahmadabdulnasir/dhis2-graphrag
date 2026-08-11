"""Hybrid retrieval: graph traversal + vector search, merged (S3.4.3).

Graph facts carry the numeric aggregates and provenance; vector chunks add
record-level context. The merged context is what the generator receives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.entity_linker import EntityLinker, LinkedQuery
from src.retrieval.graph_retriever import GraphContext, GraphRetriever
from src.retrieval.vector import VectorRetriever


@dataclass
class HybridContext:
    linked: LinkedQuery
    graph: GraphContext
    chunks: list = field(default_factory=list)   # [(Chunk, score)]

    @property
    def record_ids(self) -> list[str]:
        rids = list(self.graph.record_ids)
        for chunk, _ in self.chunks:
            rids.extend(chunk.record_ids)
        return rids


class HybridRetriever:
    def __init__(self, graph_retriever: GraphRetriever,
                 vector_retriever: VectorRetriever, linker: EntityLinker):
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.linker = linker

    def retrieve(self, question: str, k: int = 5) -> HybridContext:
        lq = self.linker.link(question)
        graph_ctx = self.graph_retriever.retrieve(lq)
        chunks = self.vector_retriever.retrieve(question, k=k)
        return HybridContext(linked=lq, graph=graph_ctx, chunks=chunks)
