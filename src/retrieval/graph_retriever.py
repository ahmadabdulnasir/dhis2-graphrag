"""Graph traversal retrieval (the Graph side of hybrid retrieval).

Given a linked query, traverses the knowledge graph to gather grounded
facts as triplets with provenance, shaped by intent:

- aggregation:   one AggResult for the indicator/scope/period
- comparison:    AggResult per LGA or state, ranked
- trend:         AggResult per year
- relationship:  AggResult per indicator within a region, ranked
- ambiguous:     candidate interpretations instead of a guess
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.graph.store import AggResult, GraphStore
from src.retrieval.entity_linker import LinkedQuery


@dataclass
class GraphContext:
    intent: str
    facts: list[AggResult] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)  # (name, total)
    series: list[tuple[str, float]] = field(default_factory=list)   # (year, total)
    candidates: list[str] = field(default_factory=list)             # for ambiguous
    notes: list[str] = field(default_factory=list)

    @property
    def triplets(self) -> list[str]:
        return [f.triplet() for f in self.facts]

    @property
    def record_ids(self) -> list[str]:
        out = []
        for f in self.facts:
            out.extend(f.record_ids)
        return out


class GraphRetriever:
    def __init__(self, store: GraphStore):
        self.store = store

    def retrieve(self, lq: LinkedQuery) -> GraphContext:
        handler = {
            "aggregation": self._aggregation,
            "comparison": self._comparison,
            "trend": self._trend,
            "relationship": self._relationship,
            "ambiguous": self._ambiguous,
        }[lq.intent]
        return handler(lq)

    # ---------- handlers ----------
    def _period(self, lq: LinkedQuery) -> str | None:
        return lq.years[0] if len(lq.years) == 1 else None

    def _aggregation(self, lq: LinkedQuery) -> GraphContext:
        ctx = GraphContext(intent="aggregation")
        res = self.store.aggregate(lq.indicator_id, state=lq.state, lga=lq.lga,
                                   region=lq.region, period_prefix=self._period(lq))
        ctx.facts.append(res)
        return ctx

    def _comparison(self, lq: LinkedQuery) -> GraphContext:
        ctx = GraphContext(intent="comparison")
        group_by = "lga" if "lga" in lq.question.lower() else "state"
        results = self.store.compare(lq.indicator_id, group_by=group_by,
                                     period_prefix=self._period(lq), state=lq.state)
        ranked = sorted(results.items(), key=lambda kv: kv[1].total, reverse=True)
        ctx.ranking = [(name, r.total) for name, r in ranked]
        ctx.facts = [r for _, r in ranked[:5]]
        return ctx

    def _trend(self, lq: LinkedQuery) -> GraphContext:
        ctx = GraphContext(intent="trend")
        years = lq.years or [str(y) for y in range(2020, 2026)]
        results = self.store.trend(lq.indicator_id, lq.state, years)
        ctx.series = [(y, r.total) for y, r in sorted(results.items())]
        ctx.facts = list(results.values())
        return ctx

    def _relationship(self, lq: LinkedQuery) -> GraphContext:
        ctx = GraphContext(intent="relationship")
        region = lq.region or "North"
        results = self.store.indicators_in_region(region, period_prefix=self._period(lq))
        # rank disease-type indicators by burden
        disease = {k: v for k, v in results.items()
                   if k in ("malaria_confirmed", "tb_notified")}
        pool = disease or results
        ranked = sorted(pool.items(), key=lambda kv: kv[1].total, reverse=True)
        ctx.ranking = [(r.indicator, r.total) for _, r in ranked]
        ctx.facts = [r for _, r in ranked]
        return ctx

    def _ambiguous(self, lq: LinkedQuery) -> GraphContext:
        """Surface candidate interpretations rather than guessing (S3.4.3)."""
        ctx = GraphContext(intent="ambiguous")
        ctx.notes.append(lq.ambiguous_reason or "under-specified question")
        names = self.store.entity_names()
        for slug, name in names["indicators"].items():
            if slug in ("malaria_confirmed", "tb_notified"):
                res = self.store.aggregate(slug, period_prefix=self._period(lq) or "2025")
                ctx.facts.append(res)
                ctx.candidates.append(name)
        return ctx
