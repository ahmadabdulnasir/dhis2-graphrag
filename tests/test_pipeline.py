"""Tests for the core pipeline: extraction, validation, graph, retrieval."""
import pandas as pd
import pytest

from src.extraction.dhis2_client import parse_analytics
from src.extraction.synthetic import generate
from src.graph.store import InMemoryGraphStore
from src.retrieval.entity_linker import EntityLinker
from src.validation.quality import validate


@pytest.fixture(scope="module")
def data():
    return generate(start="2023-01", end="2024-12")


@pytest.fixture(scope="module")
def clean(data):
    clean_df, report = validate(data)
    return clean_df, report


@pytest.fixture(scope="module")
def store(clean):
    s = InMemoryGraphStore()
    s.build(clean[0])
    return s


def test_synthetic_schema(data):
    from src.models import RECORD_COLUMNS
    assert list(data.columns) == RECORD_COLUMNS
    assert data.state.nunique() == 6
    assert data.indicator_id.nunique() == 6


def test_synthetic_deterministic():
    a = generate(start="2023-01", end="2023-03")
    b = generate(start="2023-01", end="2023-03")
    pd.testing.assert_frame_equal(a, b)


def test_validation_targets(clean):
    _, report = clean
    assert report.completeness >= 0.95
    assert report.consistency >= 0.90
    assert report.n_missing > 0          # injected problems were found
    assert report.n_outliers > 0


def test_validation_drops_logical_errors(clean):
    clean_df, _ = clean
    wide = clean_df.pivot_table(index=["facility_id", "period"],
                                columns="indicator_id", values="value")
    both = wide.dropna(subset=["anc_1st_visit", "anc_4th_visit"])
    assert (both["anc_4th_visit"] <= both["anc_1st_visit"]).all()


def test_graph_aggregation_matches_pandas(store, clean):
    """Graph traversal must equal an independent pandas aggregation."""
    clean_df, _ = clean
    expected = clean_df[(clean_df.indicator_id == "malaria_confirmed")
                        & (clean_df.state == "Kano")
                        & clean_df.period.str.startswith("2024")]["value"].sum()
    res = store.aggregate("malaria_confirmed", state="Kano", period_prefix="2024")
    assert res.total == pytest.approx(expected)
    assert res.n_values > 0
    assert len(res.record_ids) == res.n_values


def test_graph_region_traversal(store, clean):
    clean_df, _ = clean
    expected = clean_df[(clean_df.indicator_id == "malaria_confirmed")
                        & (clean_df.region == "North")]["value"].sum()
    res = store.aggregate("malaria_confirmed", region="North")
    assert res.total == pytest.approx(expected)


def test_entity_linker(store):
    linker = EntityLinker(store.entity_names())
    lq = linker.link("What was the malaria incidence in Kano State in 2024?")
    assert lq.indicator_id == "malaria_confirmed"
    assert lq.state == "Kano"
    assert lq.years == ["2024"]
    assert lq.intent == "aggregation"

    lq2 = linker.link("Which LGA had the highest maternal delivery in 2023?")
    assert lq2.intent == "comparison"

    lq3 = linker.link("Where is the current outbreak?")
    assert lq3.intent == "ambiguous"


def test_parse_analytics_fixture():
    payload = {
        "headers": [{"name": "dx"}, {"name": "ou"}, {"name": "pe"}, {"name": "value"}],
        "metaData": {"items": {
            "de1": {"name": "Malaria confirmed cases"},
            "ou1": {"name": "Some PHC"},
        }},
        "rows": [["de1", "ou1", "202403", "42"]],
    }
    df = parse_analytics(payload)
    assert len(df) == 1
    assert df.iloc[0]["value"] == 42.0
    assert df.iloc[0]["period"] == "2024-03"
    assert df.iloc[0]["indicator"] == "Malaria confirmed cases"


def test_end_to_end_answer(store, clean):
    from src.generation.answerer import OfflineAnswerer
    from src.retrieval.graph_retriever import GraphRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.vector import VectorRetriever

    clean_df, _ = clean
    linker = EntityLinker(store.entity_names())
    hybrid = HybridRetriever(GraphRetriever(store), VectorRetriever(clean_df), linker)
    ans = OfflineAnswerer().answer(
        hybrid.retrieve("How many malaria confirmed cases in Kano State in 2024?"))
    expected = clean_df[(clean_df.indicator_id == "malaria_confirmed")
                        & (clean_df.state == "Kano")
                        & clean_df.period.str.startswith("2024")]["value"].sum()
    assert f"{expected:,.0f}" in ans.text
    assert ans.record_ids  # provenance present
