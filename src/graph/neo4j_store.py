"""Neo4j implementation of GraphStore (production path).

Same interface as InMemoryGraphStore; aggregations run as Cypher queries.
Start the database with `docker compose up -d`, then pass
Neo4jGraphStore.from_env() to the pipeline.
"""
from __future__ import annotations

import os

import pandas as pd

from src.graph.store import AggResult, GraphStore


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, username: str, password: str):
        from neo4j import GraphDatabase  # imported lazily
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    @classmethod
    def from_env(cls) -> "Neo4jGraphStore":
        return cls(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "graphrag-dev"),
        )

    def close(self):
        self.driver.close()

    # ---------- construction ----------
    def build(self, df: pd.DataFrame) -> None:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            for label in ("Indicator", "Facility", "LGA", "State", "Period"):
                s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.key)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (dv:DataValue) ON (dv.indicator_id)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (dv:DataValue) ON (dv.period)")
            rows = df.to_dict("records")
            for i in range(0, len(rows), 1000):
                s.run(
                    """
                    UNWIND $rows AS r
                    MERGE (i:Indicator {key: r.indicator_id}) SET i.name = r.indicator
                    MERGE (f:Facility {key: r.facility_id}) SET f.name = r.facility
                    MERGE (l:LGA {key: r.lga}) SET l.name = r.lga
                    MERGE (st:State {key: r.state}) SET st.name = r.state
                    MERGE (rg:Region {key: r.region}) SET rg.name = r.region
                    MERGE (p:Period {key: r.period})
                    MERGE (f)-[:LOCATED_IN]->(l)
                    MERGE (l)-[:LOCATED_IN]->(st)
                    MERGE (st)-[:PART_OF]->(rg)
                    CREATE (dv:DataValue {record_id: r.record_id, value: r.value,
                                          indicator_id: r.indicator_id, period: r.period})
                    CREATE (dv)-[:OF_INDICATOR]->(i)
                    CREATE (dv)-[:REPORTED_AT]->(f)
                    CREATE (dv)-[:FOR_PERIOD]->(p)
                    """,
                    rows=rows[i:i + 1000],
                )

    # ---------- queries ----------
    _SCOPE_MATCH = {
        "lga":    "(f)-[:LOCATED_IN]->(:LGA {key: $scope})",
        "state":  "(f)-[:LOCATED_IN]->(:LGA)-[:LOCATED_IN]->(:State {key: $scope})",
        "region": "(f)-[:LOCATED_IN]->(:LGA)-[:LOCATED_IN]->(:State)-[:PART_OF]->(:Region {key: $scope})",
    }

    def _run_agg(self, indicator_id, scope_kind, scope, period_prefix) -> AggResult:
        scope_clause = f"MATCH {self._SCOPE_MATCH[scope_kind]}" if scope_kind else ""
        period_clause = "AND dv.period STARTS WITH $pp" if period_prefix else ""
        q = f"""
            MATCH (dv:DataValue)-[:REPORTED_AT]->(f:Facility)
            {scope_clause}
            WHERE dv.indicator_id = $ind {period_clause}
            RETURN sum(dv.value) AS total, count(dv) AS n,
                   collect(dv.record_id)[0..200] AS rids
        """
        with self.driver.session() as s:
            rec = s.run(q, ind=indicator_id, scope=scope, pp=period_prefix).single()
        name = self._indicator_name(indicator_id)
        scope_name = {"lga": f"{scope} LGA", "state": f"{scope} State",
                      "region": f"{scope}ern region"}.get(scope_kind, "all states")
        return AggResult(indicator=name, scope=scope_name, period=period_prefix or "all",
                         total=rec["total"] or 0.0, n_values=rec["n"],
                         record_ids=rec["rids"])

    def _indicator_name(self, indicator_id) -> str:
        with self.driver.session() as s:
            rec = s.run("MATCH (i:Indicator {key: $k}) RETURN i.name AS n",
                        k=indicator_id).single()
        return rec["n"] if rec else indicator_id

    def aggregate(self, indicator_id, state=None, lga=None, region=None,
                  period_prefix=None) -> AggResult:
        if lga:
            return self._run_agg(indicator_id, "lga", lga, period_prefix)
        if state:
            return self._run_agg(indicator_id, "state", state, period_prefix)
        if region:
            return self._run_agg(indicator_id, "region", region, period_prefix)
        return self._run_agg(indicator_id, None, None, period_prefix)

    def compare(self, indicator_id, group_by, period_prefix=None, state=None) -> dict:
        label = {"lga": "LGA", "state": "State"}[group_by]
        with self.driver.session() as s:
            if group_by == "lga" and state:
                names = [r["k"] for r in s.run(
                    "MATCH (l:LGA)-[:LOCATED_IN]->(:State {key: $st}) RETURN l.key AS k",
                    st=state)]
            else:
                names = [r["k"] for r in s.run(f"MATCH (n:{label}) RETURN n.key AS k")]
        kw = "lga" if group_by == "lga" else "state"
        return {n: self.aggregate(indicator_id, **{kw: n}, period_prefix=period_prefix)
                for n in names}

    def trend(self, indicator_id, state, years) -> dict:
        return {y: self.aggregate(indicator_id, state=state, period_prefix=y) for y in years}

    def indicators_in_region(self, region, period_prefix=None) -> dict:
        with self.driver.session() as s:
            slugs = [r["k"] for r in s.run("MATCH (i:Indicator) RETURN i.key AS k")]
        return {sl: self.aggregate(sl, region=region, period_prefix=period_prefix)
                for sl in slugs}

    def entity_names(self) -> dict:
        with self.driver.session() as s:
            get = lambda lbl: sorted(r["n"] for r in s.run(
                f"MATCH (x:{lbl}) RETURN coalesce(x.name, x.key) AS n"))
            inds = {r["k"]: r["n"] for r in s.run(
                "MATCH (i:Indicator) RETURN i.key AS k, i.name AS n")}
            return {"states": get("State"), "lgas": get("LGA"),
                    "regions": get("Region"), "indicators": inds}
