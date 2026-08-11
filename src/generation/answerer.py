"""Grounded generation with provenance (pipeline stage 4).

Two modes:
- OfflineAnswerer: deterministic templates over graph facts. No API needed,
  fully reproducible — used for development and the offline evaluation runs.
- LLMAnswerer: sends retrieved triplets/records to an LLM via API with an
  instruction to answer only from the supplied context. Used when
  OPENAI_API_KEY is set.

Every Answer carries the supporting triplets and record ids (provenance),
per the explainability requirement in S3.2.2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.retrieval.hybrid import HybridContext


@dataclass
class Answer:
    question: str
    text: str
    triplets: list = field(default_factory=list)
    record_ids: list = field(default_factory=list)
    n_supporting_values: int = 0
    context_chars: int = 0        # size of retrieved context (Experiment B)
    contexts: list = field(default_factory=list)  # the retrieved context itself
    mode: str = "offline"

    def provenance_summary(self) -> str:
        return (f"grounded in {self.n_supporting_values} data values; "
                f"{len(self.triplets)} triplets; "
                f"records e.g. {self.record_ids[:3]}")


def context_lines(ctx: HybridContext) -> list[str]:
    """The retrieved context as text: graph facts first, then passages.

    Shared by both answer modes and by the evaluation (RAGAS contexts,
    hallucination check), so what is measured is exactly what was retrieved.
    """
    g = ctx.graph
    lines = list(g.triplets)
    if g.ranking:
        lines.append("Ranking: " + "; ".join(f"{n}: {t:,.0f}" for n, t in g.ranking))
    if g.series:
        lines.append("Yearly series: " + "; ".join(f"{y}: {v:,.0f}" for y, v in g.series))
    lines += [c.text for c, _ in ctx.chunks]
    return lines


class OfflineAnswerer:
    """Composes answers directly from graph facts. Deterministic."""

    def answer(self, ctx: HybridContext) -> Answer:
        g = ctx.graph
        text = {
            "aggregation": self._aggregation,
            "comparison": self._comparison,
            "trend": self._trend,
            "relationship": self._relationship,
            "ambiguous": self._ambiguous,
        }[g.intent](ctx)
        rids = g.record_ids
        lines = context_lines(ctx)
        return Answer(question=ctx.linked.question, text=text,
                      triplets=g.triplets, record_ids=rids[:200],
                      n_supporting_values=sum(f.n_values for f in g.facts),
                      context_chars=sum(len(t) for t in g.triplets),
                      contexts=lines, mode="offline")

    def _aggregation(self, ctx) -> str:
        f = ctx.graph.facts[0]
        if f.n_values == 0:
            return (f"No reported values were found for {f.indicator} in "
                    f"{f.scope} for period {f.period}.")
        return (f"{f.indicator} in {f.scope} for {f.period}: total "
                f"{f.total:,.0f}, from {f.n_values} facility-month reports.")

    def _comparison(self, ctx) -> str:
        r = ctx.graph.ranking
        if not r:
            return "No data found to compare."
        top = r[0]
        f = ctx.graph.facts[0]
        word = ctx.graph.direction
        lines = ", ".join(f"{name}: {total:,.0f}" for name, total in r[:5])
        return (f"{top[0]} had the {word} {f.indicator} "
                f"({top[1]:,.0f}) for period {f.period}. Ranked results — {lines}.")

    def _trend(self, ctx) -> str:
        s = ctx.graph.series
        if not s:
            return "No data found for the requested period."
        f = ctx.graph.facts[0]
        direction = "increased" if s[-1][1] > s[0][1] else "decreased"
        pct = abs(s[-1][1] - s[0][1]) / s[0][1] * 100 if s[0][1] else 0.0
        series_txt = "; ".join(f"{y}: {v:,.0f}" for y, v in s)
        return (f"{f.indicator} in {f.scope} {direction} by {pct:.0f}% "
                f"between {s[0][0]} and {s[-1][0]}. Yearly totals — {series_txt}.")

    def _relationship(self, ctx) -> str:
        r = ctx.graph.ranking
        if not r:
            return "No data found."
        lines = "; ".join(f"{name}: {total:,.0f}" for name, total in r)
        return (f"By reported burden, the most prevalent is {r[0][0]} "
                f"({r[0][1]:,.0f} cases). Ranking — {lines}.")

    def _ambiguous(self, ctx) -> str:
        g = ctx.graph
        cands = ", ".join(g.candidates) or "the tracked disease indicators"
        note = g.notes[0] if g.notes else "the question is under-specified"
        parts = "; ".join(f"{f.indicator}: {f.total:,.0f} ({f.period})" for f in g.facts)
        return (f"The question is ambiguous ({note}). Rather than guess, here are "
                f"current totals for {cands} so you can specify which one you mean"
                + (f" — {parts}." if parts else "."))


class LLMAnswerer:
    """LLM generation grounded in the retrieved context."""

    SYSTEM = (
        "You are a public health data assistant. Answer ONLY from the facts and "
        "records supplied in the context. If the context does not contain the "
        "answer, say so. Always include the key numbers. Never invent values."
    )

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, ctx: HybridContext) -> Answer:
        g = ctx.graph
        lines = context_lines(ctx)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": "Context:\n" + "\n".join(lines)
                                            + f"\n\nQuestion: {ctx.linked.question}"},
            ],
            temperature=0,
        )
        rids = ctx.record_ids
        return Answer(question=ctx.linked.question,
                      text=resp.choices[0].message.content.strip(),
                      triplets=g.triplets, record_ids=rids[:200],
                      n_supporting_values=sum(f.n_values for f in g.facts),
                      context_chars=sum(len(l) for l in lines),
                      contexts=lines, mode="llm")


def default_answerer():
    if os.getenv("OPENAI_API_KEY"):
        try:
            return LLMAnswerer()
        except Exception:
            pass
    return OfflineAnswerer()
