"""Entity linking: map a natural-language question onto graph entities.

Identifies the indicator (via name + synonym matching), geographic scope
(state, LGA, region), time scope (years), and the query intent
(aggregation, comparison, trend, relationship, ambiguous).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models import INDICATOR_SYNONYMS


@dataclass
class LinkedQuery:
    question: str
    indicator_id: str | None = None
    state: str | None = None
    lga: str | None = None
    region: str | None = None
    years: list[str] = field(default_factory=list)
    intent: str = "aggregation"
    direction: str = "highest"      # "highest" | "lowest" (for comparisons)
    ambiguous_reason: str | None = None


class EntityLinker:
    def __init__(self, entity_names: dict):
        self.states = entity_names["states"]
        self.lgas = entity_names["lgas"]
        self.indicators = entity_names["indicators"]  # slug -> display name

    def link(self, question: str) -> LinkedQuery:
        q = question.lower()
        lq = LinkedQuery(question=question)

        # indicator: exact display name first, then synonyms (longest match wins)
        best_len = 0
        for slug, name in self.indicators.items():
            candidates = [name.lower()] + INDICATOR_SYNONYMS.get(slug, [])
            for c in candidates:
                # lookarounds instead of \b so names ending in ')' or '+'
                # (e.g. DHS "ANC 4+ visits (%)") still match
                if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", q) and len(c) > best_len:
                    lq.indicator_id, best_len = slug, len(c)

        # geography
        for s in self.states:
            if s.lower() in q:
                lq.state = s
                break
        for l in self.lgas:
            if l.lower() in q:
                lq.lga = l
                break
        if lq.lga and lq.lga == lq.state:
            lq.lga = None  # state-level datasets mirror states as LGAs
        if re.search(r"\bnorth(ern)?\b", q):
            lq.region = "North"
        elif re.search(r"\bsouth(ern)?\b", q):
            lq.region = "South"

        # time: explicit years, or "last N years"
        lq.years = sorted(set(re.findall(r"\b(20\d{2})\b", q)))
        m = re.search(r"last (\w+|\d+) years?", q)
        if m and not lq.years:
            words = {"two": 2, "three": 3, "four": 4, "five": 5}
            n = words.get(m.group(1), None) or (int(m.group(1)) if m.group(1).isdigit() else 5)
            lq.years = [str(y) for y in range(2025 - n + 1, 2026)]

        if re.search(r"\b(lowest|least|fewest|worst|bottom)\b", q):
            lq.direction = "lowest"

        # intent
        if re.search(r"\b(trend|changed|change|over the last|over time|evolution)\b", q):
            lq.intent = "trend"
        elif re.search(r"\b(which|highest|lowest|most|least|best|worst|compare|comparison|top)\b", q) \
                and re.search(r"\b(lga|state|facility|region|diseases?|indicators?)\b", q):
            lq.intent = "relationship" if re.search(r"\bdiseases?|indicators?\b", q) and lq.region \
                else "comparison"
        elif re.search(r"\b(outbreak|situation|problem)\b", q) and not lq.indicator_id:
            lq.intent = "ambiguous"
            lq.ambiguous_reason = "no disease or indicator specified"
        if lq.indicator_id is None and lq.intent not in ("relationship", "ambiguous"):
            lq.intent = "ambiguous"
            lq.ambiguous_reason = lq.ambiguous_reason or "could not identify an indicator"
        return lq
