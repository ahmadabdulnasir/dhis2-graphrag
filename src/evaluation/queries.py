"""Evaluation query set (S3.5) — five categories, 36 queries.

Ground truth is computed *directly from the validated dataframe with pandas*,
independently of the graph traversal code, so the evaluation doubles as a
correctness check on the graph store. For the real-data track, expert
verification replaces these programmatic gold answers.

Categories and counts:
    aggregation 10 | comparison 8 | trend 6 | relationship 6 | ambiguous 6
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


# ---------- gold helpers (pandas only, no graph code) ----------
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


def _rank_group(df, ind, by, year=None, state=None, ascending=False):
    m = df.indicator_id == ind
    if year:
        m &= df.period.str.startswith(year)
    if state:
        m &= df.state == state
    s = df.loc[m].groupby(by)["value"].sum().sort_values(ascending=ascending)
    if len(s) == 0:  # dataset does not contain this indicator/scope
        return None, None
    return s.index[0], float(s.iloc[0])


def _agg(ind, **kw):
    return lambda df: (_total(df, ind, **kw), None)


def _cmp(ind, by, ascending=False, **kw):
    return lambda df: (None, _rank_group(df, ind, by, ascending=ascending, **kw)[0])


def _trend(ind, state, y0, y1):
    def fn(df):
        a, b = _total(df, ind, state=state, year=y0), _total(df, ind, state=state, year=y1)
        if a == 0 and b == 0:  # indicator/scope absent from this dataset
            return None, None
        return None, ("increased" if b > a else "decreased")
    return fn


def _rel(region, year=None):
    def fn(df):
        m = _total(df, "malaria_confirmed", region=region, year=year)
        t = _total(df, "tb_notified", region=region, year=year)
        if m == 0 and t == 0:  # indicators absent from this dataset
            return None, None
        return None, ("Malaria confirmed cases" if m >= t else "TB cases notified")
    return fn


_AMBIGUOUS = lambda df: (None, "ambiguous")


def build_queries() -> list[EvalQuery]:
    Q = [
        # ---- aggregation (10) ----
        EvalQuery("agg-01", "aggregation",
                  "What was the total number of malaria confirmed cases in Kano State in 2024?",
                  _agg("malaria_confirmed", state="Kano", year="2024")),
        EvalQuery("agg-02", "aggregation",
                  "How many facility deliveries were reported in Lagos in 2023?",
                  _agg("facility_delivery", state="Lagos", year="2023")),
        EvalQuery("agg-03", "aggregation",
                  "Total TB cases notified in Borno State in 2022?",
                  _agg("tb_notified", state="Borno", year="2022")),
        EvalQuery("agg-04", "aggregation",
                  "How many ANC 1st visit contacts were recorded in Ikeja LGA in 2024?",
                  _agg("anc_1st_visit", lga="Ikeja", year="2024")),
        EvalQuery("agg-05", "aggregation",
                  "How many Penta 3 doses given were recorded nationwide in 2025?",
                  _agg("penta3", year="2025")),
        EvalQuery("agg-06", "aggregation",
                  "What was the malaria confirmed cases burden in northern Nigeria in 2023?",
                  _agg("malaria_confirmed", region="North", year="2023")),
        EvalQuery("agg-07", "aggregation",
                  "How many ANC 4th visit contacts were reported in Rivers State in 2024?",
                  _agg("anc_4th_visit", state="Rivers", year="2024")),
        EvalQuery("agg-08", "aggregation",
                  "Total facility deliveries in Zaria LGA in 2022?",
                  _agg("facility_delivery", lga="Zaria", year="2022")),
        EvalQuery("agg-09", "aggregation",
                  "How many TB cases notified were recorded in southern Nigeria in 2024?",
                  _agg("tb_notified", region="South", year="2024")),
        EvalQuery("agg-10", "aggregation",
                  "What is the total number of malaria confirmed cases recorded in Kano State?",
                  _agg("malaria_confirmed", state="Kano")),

        # ---- comparison (8) ----
        EvalQuery("cmp-01", "comparison",
                  "Which state had the highest malaria confirmed cases in 2024?",
                  _cmp("malaria_confirmed", "state", year="2024")),
        EvalQuery("cmp-02", "comparison",
                  "Which LGA in Kano had the most facility deliveries in 2023?",
                  _cmp("facility_delivery", "lga", year="2023", state="Kano")),
        EvalQuery("cmp-03", "comparison",
                  "Which state recorded the highest Penta 3 doses given in 2025?",
                  _cmp("penta3", "state", year="2025")),
        EvalQuery("cmp-04", "comparison",
                  "Which LGA had the highest TB cases notified in 2024?",
                  _cmp("tb_notified", "lga", year="2024")),
        EvalQuery("cmp-05", "comparison",
                  "Which state had the lowest facility deliveries in 2024?",
                  _cmp("facility_delivery", "state", year="2024", ascending=True)),
        EvalQuery("cmp-06", "comparison",
                  "Which LGA in Lagos had the highest malaria confirmed cases in 2025?",
                  _cmp("malaria_confirmed", "lga", year="2025", state="Lagos")),
        EvalQuery("cmp-07", "comparison",
                  "Which state had the most ANC 1st visit contacts in 2023?",
                  _cmp("anc_1st_visit", "state", year="2023")),
        EvalQuery("cmp-08", "comparison",
                  "Which LGA recorded the lowest Penta 3 doses given in 2022?",
                  _cmp("penta3", "lga", year="2022", ascending=True)),

        # ---- trend (6) ----
        EvalQuery("trd-01", "trend",
                  "How has immunisation coverage changed in Lagos State over the last five years?",
                  _trend("penta3", "Lagos", "2021", "2025")),
        EvalQuery("trd-02", "trend",
                  "What is the trend of malaria cases in Kano between 2020 and 2025?",
                  _trend("malaria_confirmed", "Kano", "2020", "2025")),
        EvalQuery("trd-03", "trend",
                  "How have facility deliveries changed in Rivers State over time?",
                  _trend("facility_delivery", "Rivers", "2020", "2025")),
        EvalQuery("trd-04", "trend",
                  "What is the trend of TB cases notified in Borno between 2021 and 2025?",
                  _trend("tb_notified", "Borno", "2021", "2025")),
        EvalQuery("trd-05", "trend",
                  "How have ANC 1st visit contacts changed in Enugu over the last three years?",
                  _trend("anc_1st_visit", "Enugu", "2023", "2025")),
        EvalQuery("trd-06", "trend",
                  "How has Penta 3 doses given changed in Kaduna between 2020 and 2024?",
                  _trend("penta3", "Kaduna", "2020", "2024")),

        # ---- relationship (6) ----
        EvalQuery("rel-01", "relationship",
                  "Which diseases are most prevalent in northern Nigeria?", _rel("North")),
        EvalQuery("rel-02", "relationship",
                  "Which diseases are most prevalent in southern Nigeria?", _rel("South")),
        EvalQuery("rel-03", "relationship",
                  "Which diseases had the highest burden in the north in 2024?",
                  _rel("North", "2024")),
        EvalQuery("rel-04", "relationship",
                  "Which diseases were most reported in the south in 2022?",
                  _rel("South", "2022")),
        EvalQuery("rel-05", "relationship",
                  "Which disease had the highest burden in northern Nigeria in 2021?",
                  _rel("North", "2021")),
        EvalQuery("rel-06", "relationship",
                  "What are the top diseases in the south in 2025?", _rel("South", "2025")),

        # ---- ambiguous (6): correct behaviour = flag ambiguity, don't guess ----
        EvalQuery("amb-01", "ambiguous", "Where is the current outbreak?", _AMBIGUOUS),
        EvalQuery("amb-02", "ambiguous", "What is the situation this month?", _AMBIGUOUS),
        EvalQuery("amb-03", "ambiguous", "Where is the outbreak happening right now?", _AMBIGUOUS),
        EvalQuery("amb-04", "ambiguous", "Is there a problem in the north?", _AMBIGUOUS),
        EvalQuery("amb-05", "ambiguous", "What should we be worried about?", _AMBIGUOUS),
        EvalQuery("amb-06", "ambiguous", "Give me the numbers.", _AMBIGUOUS),
    ]
    return Q


def gold_answers(df: pd.DataFrame) -> dict:
    return {q.qid: q.gold_fn(df) for q in build_queries()}
