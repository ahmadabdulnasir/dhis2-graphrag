"""Evaluation query set (S3.5) — five categories, starter set of 20.

Ground truth is computed *directly from the validated dataframe with pandas*,
independently of the graph traversal code, so the evaluation doubles as a
correctness check on the graph store. Expand to 30-50 queries before the
final experiments (expert verification then replaces the programmatic gold
answers for the real-data track).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class EvalQuery:
    qid: str
    category: str          # aggregation | comparison | trend | relationship | ambiguous
    question: str
    gold_fn: Callable      # df -> (gold_number | None, gold_text)


def _total(df, ind, state=None, lga=None, region=None, year=None) -> float:
    m = df.indicator_id == ind
    if state:
        m &= df.state == state
    if lga:
        m &= df.lga == lga
    if region:
        m &= df.region == region
    if year:
        m &= df.period.str.startswith(year)
    return float(df.loc[m, "value"].sum())


def _top_group(df, ind, by, year=None, state=None):
    m = df.indicator_id == ind
    if year:
        m &= df.period.str.startswith(year)
    if state:
        m &= df.state == state
    s = df.loc[m].groupby(by)["value"].sum().sort_values(ascending=False)
    return s.index[0], float(s.iloc[0])


def build_queries() -> list[EvalQuery]:
    Q = []

    # ---- aggregation ----
    Q.append(EvalQuery("agg-01", "aggregation",
        "What was the total number of malaria confirmed cases in Kano State in 2024?",
        lambda df: (_total(df, "malaria_confirmed", state="Kano", year="2024"), None)))
    Q.append(EvalQuery("agg-02", "aggregation",
        "How many facility deliveries were reported in Lagos in 2023?",
        lambda df: (_total(df, "facility_delivery", state="Lagos", year="2023"), None)))
    Q.append(EvalQuery("agg-03", "aggregation",
        "Total TB cases notified in Borno State in 2022?",
        lambda df: (_total(df, "tb_notified", state="Borno", year="2022"), None)))
    Q.append(EvalQuery("agg-04", "aggregation",
        "How many ANC 1st visit contacts were recorded in Ikeja LGA in 2024?",
        lambda df: (_total(df, "anc_1st_visit", lga="Ikeja", year="2024"), None)))

    # ---- comparison ----
    Q.append(EvalQuery("cmp-01", "comparison",
        "Which state had the highest malaria confirmed cases in 2024?",
        lambda df: (None, _top_group(df, "malaria_confirmed", "state", year="2024")[0])))
    Q.append(EvalQuery("cmp-02", "comparison",
        "Which LGA in Kano had the most facility deliveries in 2023?",
        lambda df: (None, _top_group(df, "facility_delivery", "lga",
                                     year="2023", state="Kano")[0])))
    Q.append(EvalQuery("cmp-03", "comparison",
        "Which state recorded the highest Penta 3 doses given in 2025?",
        lambda df: (None, _top_group(df, "penta3", "state", year="2025")[0])))
    Q.append(EvalQuery("cmp-04", "comparison",
        "Which LGA had the highest TB cases notified in 2024?",
        lambda df: (None, _top_group(df, "tb_notified", "lga", year="2024")[0])))

    # ---- trend ----
    def trend_gold(ind, state):
        def fn(df):
            y0 = _total(df, ind, state=state, year="2020")
            y1 = _total(df, ind, state=state, year="2025")
            direction = "increased" if y1 > y0 else "decreased"
            return None, direction
        return fn
    Q.append(EvalQuery("trd-01", "trend",
        "How has immunisation coverage changed in Lagos State over the last five years?",
        trend_gold("penta3", "Lagos")))
    Q.append(EvalQuery("trd-02", "trend",
        "What is the trend of malaria cases in Kano between 2020 and 2025?",
        trend_gold("malaria_confirmed", "Kano")))
    Q.append(EvalQuery("trd-03", "trend",
        "How have facility deliveries changed in Rivers State over time?",
        trend_gold("facility_delivery", "Rivers")))

    # ---- relationship ----
    def rel_gold(region):
        def fn(df):
            m = _total(df, "malaria_confirmed", region=region)
            t = _total(df, "tb_notified", region=region)
            name = "Malaria confirmed cases" if m >= t else "TB cases notified"
            return None, name
        return fn
    Q.append(EvalQuery("rel-01", "relationship",
        "Which diseases are most prevalent in northern Nigeria?", rel_gold("North")))
    Q.append(EvalQuery("rel-02", "relationship",
        "Which diseases are most prevalent in southern Nigeria?", rel_gold("South")))

    # ---- ambiguous ----
    # gold behaviour: system must flag ambiguity / ask for the disease,
    # not invent an answer.
    Q.append(EvalQuery("amb-01", "ambiguous",
        "Where is the current outbreak?",
        lambda df: (None, "ambiguous")))
    Q.append(EvalQuery("amb-02", "ambiguous",
        "What is the situation this month?",
        lambda df: (None, "ambiguous")))

    return Q


def gold_answers(df: pd.DataFrame) -> dict:
    return {q.qid: q.gold_fn(df) for q in build_queries()}
