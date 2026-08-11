"""GraphStore interface + in-memory implementation.

The retriever talks to a GraphStore, never to a database directly, so the
same pipeline runs against Neo4j (production, see neo4j_store.py) or the
in-memory networkx store (development, tests, offline evaluation).

Aggregations are computed by *graph traversal* — walking LOCATED_IN /
PART_OF edges down to facilities and REPORTED_AT edges to DataValue nodes —
not by dataframe operations. This keeps the in-memory store faithful to how
the Cypher queries behave on Neo4j.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from src.graph.schema import node_id


@dataclass
class AggResult:
    """A grounded numeric fact with provenance."""
    indicator: str
    scope: str                    # e.g. "Kano State", "northern region"
    period: str                   # e.g. "2024", "2023-01..2023-12"
    total: float = 0.0
    n_values: int = 0
    record_ids: list = field(default_factory=list)

    def triplet(self) -> str:
        return f"[{self.indicator}] -[OCCURRED_IN]-> [{self.scope}] (period {self.period}, total {self.total:,.0f})"


class GraphStore(ABC):
    @abstractmethod
    def build(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def aggregate(self, indicator_id: str, state: str | None = None,
                  lga: str | None = None, region: str | None = None,
                  period_prefix: str | None = None) -> AggResult: ...

    @abstractmethod
    def compare(self, indicator_id: str, group_by: str,
                period_prefix: str | None = None,
                state: str | None = None) -> dict[str, AggResult]: ...

    @abstractmethod
    def trend(self, indicator_id: str, state: str | None,
              years: list[str]) -> dict[str, AggResult]: ...

    @abstractmethod
    def indicators_in_region(self, region: str,
                             period_prefix: str | None = None) -> dict[str, AggResult]: ...

    @abstractmethod
    def entity_names(self) -> dict: ...


class InMemoryGraphStore(GraphStore):
    """networkx MultiDiGraph implementation."""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._indicator_names: dict[str, str] = {}

    # ---------- construction ----------
    def build(self, df: pd.DataFrame) -> None:
        g = self.g
        for row in df.itertuples(index=False):
            ind = node_id("Indicator", row.indicator_id)
            fac = node_id("Facility", row.facility_id)
            lga = node_id("LGA", row.lga)
            state = node_id("State", row.state)
            region = node_id("Region", row.region)
            period = node_id("Period", row.period)
            dv = node_id("DataValue", row.record_id)

            # nodes first (add_edge would otherwise create attr-less nodes)
            if ind not in g:
                g.add_node(ind, label="Indicator", name=row.indicator, slug=row.indicator_id)
                self._indicator_names[row.indicator_id] = row.indicator
            if region not in g:
                g.add_node(region, label="Region", name=row.region)
            if state not in g:
                g.add_node(state, label="State", name=row.state)
                g.add_edge(state, region, key="PART_OF")
            if lga not in g:
                g.add_node(lga, label="LGA", name=row.lga)
                g.add_edge(lga, state, key="LOCATED_IN")
            if fac not in g:
                g.add_node(fac, label="Facility", name=row.facility)
                g.add_edge(fac, lga, key="LOCATED_IN")
            if period not in g:
                g.add_node(period, label="Period", name=row.period)

            g.add_node(dv, label="DataValue", value=float(row.value),
                       indicator_id=row.indicator_id, period=row.period,
                       record_id=row.record_id)
            g.add_edge(dv, ind, key="OF_INDICATOR")
            g.add_edge(dv, fac, key="REPORTED_AT")
            g.add_edge(dv, period, key="FOR_PERIOD")

    # ---------- traversal helpers ----------
    def _facilities_under(self, node: str) -> list[str]:
        """Walk LOCATED_IN / PART_OF edges backwards to reach facilities."""
        if node not in self.g:
            return []
        if self.g.nodes[node]["label"] == "Facility":
            return [node]
        out = []
        for pred, _, key in self.g.in_edges(node, keys=True):
            if key in ("LOCATED_IN", "PART_OF"):
                out.extend(self._facilities_under(pred))
        return out

    def _values_at(self, facility: str, indicator_id: str,
                   period_prefix: str | None) -> tuple[float, int, list[str]]:
        total, n, rids = 0.0, 0, []
        for dv, _, key in self.g.in_edges(facility, keys=True):
            if key != "REPORTED_AT":
                continue
            attrs = self.g.nodes[dv]
            if attrs["indicator_id"] != indicator_id:
                continue
            if period_prefix and not attrs["period"].startswith(period_prefix):
                continue
            total += attrs["value"]
            n += 1
            rids.append(attrs["record_id"])
        return total, n, rids

    def _scope_node(self, state=None, lga=None, region=None) -> tuple[str | None, str]:
        if lga:
            return node_id("LGA", lga), f"{lga} LGA"
        if state:
            return node_id("State", state), f"{state} State"
        if region:
            return node_id("Region", region), f"{region}ern region"
        return None, "all states"

    # ---------- interface ----------
    def aggregate(self, indicator_id, state=None, lga=None, region=None,
                  period_prefix=None) -> AggResult:
        scope_node, scope_name = self._scope_node(state, lga, region)
        if scope_node is None:
            facilities = [n for n, d in self.g.nodes(data=True) if d["label"] == "Facility"]
        else:
            facilities = self._facilities_under(scope_node)
        res = AggResult(indicator=self._indicator_names.get(indicator_id, indicator_id),
                        scope=scope_name, period=period_prefix or "all")
        for fac in facilities:
            t, n, rids = self._values_at(fac, indicator_id, period_prefix)
            res.total += t
            res.n_values += n
            res.record_ids.extend(rids)
        return res

    def compare(self, indicator_id, group_by, period_prefix=None, state=None) -> dict:
        label = {"lga": "LGA", "state": "State"}[group_by]
        groups = [d["name"] for n, d in self.g.nodes(data=True) if d["label"] == label]
        out = {}
        for name in groups:
            if group_by == "lga":
                # honour optional state filter by checking the LGA's parent
                if state:
                    parents = [v for _, v, k in self.g.out_edges(node_id("LGA", name), keys=True)
                               if k == "LOCATED_IN"]
                    if node_id("State", state) not in parents:
                        continue
                out[name] = self.aggregate(indicator_id, lga=name, period_prefix=period_prefix)
            else:
                out[name] = self.aggregate(indicator_id, state=name, period_prefix=period_prefix)
        return out

    def trend(self, indicator_id, state, years) -> dict:
        return {y: self.aggregate(indicator_id, state=state, period_prefix=y) for y in years}

    def indicators_in_region(self, region, period_prefix=None) -> dict:
        return {slug: self.aggregate(slug, region=region, period_prefix=period_prefix)
                for slug in self._indicator_names}

    def entity_names(self) -> dict:
        by = lambda lbl: sorted({d["name"] for _, d in self.g.nodes(data=True) if d["label"] == lbl})
        return {"states": by("State"), "lgas": by("LGA"), "regions": by("Region"),
                "indicators": dict(self._indicator_names)}

    def stats(self) -> dict:
        labels = {}
        for _, d in self.g.nodes(data=True):
            labels[d["label"]] = labels.get(d["label"], 0) + 1
        return {"nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(), "by_label": labels}
