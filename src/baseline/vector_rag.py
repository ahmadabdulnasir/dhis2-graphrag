"""Vector-only RAG baseline (the comparison system in the experiment).

Shares the vector retriever and chunking with the hybrid pipeline but has
no knowledge graph: it must answer from the top-k retrieved passages alone.
This is exactly the configuration the write-up predicts will fail on
aggregation/relationship queries, since the answer is spread across many
records while only k passages fit in the context.

Offline mode parses indicator values out of the retrieved passages and
sums them — a best-effort answer from what retrieval surfaced. LLM mode
passes the passages to the model with the same grounding instruction as
the GraphRAG system, so the only difference between systems is retrieval.
"""
from __future__ import annotations

import os
import re

from src.generation.answerer import Answer
from src.retrieval.entity_linker import EntityLinker
from src.retrieval.vector import VectorRetriever


class VectorOnlyRAG:
    def __init__(self, vector_retriever: VectorRetriever, linker: EntityLinker,
                 k: int = 8):
        self.retriever = vector_retriever
        self.linker = linker
        self.k = k
        self.use_llm = bool(os.getenv("OPENAI_API_KEY"))

    def answer(self, question: str) -> Answer:
        chunks = self.retriever.retrieve(question, k=self.k)
        if self.use_llm:
            try:
                return self._answer_llm(question, chunks)
            except Exception:
                pass
        return self._answer_offline(question, chunks)

    # ---------- offline ----------
    def _answer_offline(self, question: str, chunks) -> Answer:
        lq = self.linker.link(question)
        rids = [r for c, _ in chunks for r in c.record_ids]
        indicator_name = None
        if lq.indicator_id:
            indicator_name = self.linker.indicators.get(lq.indicator_id)

        total, n = 0.0, 0
        if indicator_name:
            pattern = re.compile(re.escape(indicator_name) + r": ([\d,]+)")
            for chunk, _ in chunks:
                for m in pattern.finditer(chunk.text):
                    total += float(m.group(1).replace(",", ""))
                    n += 1
        if n:
            text = (f"Based on the {len(chunks)} most similar records retrieved, "
                    f"{indicator_name} totals {total:,.0f} across {n} "
                    f"facility-month reports (note: retrieval may not cover all "
                    f"relevant records).")
        else:
            top = chunks[0][0].text if chunks else "no records retrieved"
            text = f"Most relevant record found: {top}"
        return Answer(question=question, text=text,
                      triplets=[], record_ids=rids[:200],
                      n_supporting_values=n, mode="baseline-offline")

    # ---------- llm ----------
    def _answer_llm(self, question: str, chunks) -> Answer:
        from openai import OpenAI
        client = OpenAI()
        context = "\n".join(c.text for c, _ in chunks)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content":
                    "You are a public health data assistant. Answer ONLY from the "
                    "records supplied. If they are insufficient, say so. Never "
                    "invent values."},
                {"role": "user", "content": f"Records:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        rids = [r for c, _ in chunks for r in c.record_ids]
        return Answer(question=question, text=resp.choices[0].message.content.strip(),
                      triplets=[], record_ids=rids[:200],
                      n_supporting_values=len(chunks), mode="baseline-llm")
