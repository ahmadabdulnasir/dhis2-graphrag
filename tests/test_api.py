"""API tests (uses a small synthetic dataset via env override)."""
import os

import pytest
from fastapi.testclient import TestClient

# force the offline configuration regardless of any local .env
os.environ["GRAPHRAG_SOURCE"] = "synthetic"
os.environ["GRAPHRAG_NEO4J"] = "0"

from src.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["validation_targets_met"] is True


def test_entities(client):
    r = client.get("/entities")
    assert r.status_code == 200
    assert "Kano" in r.json()["states"]


def test_query_aggregation(client):
    r = client.post("/query", json={
        "question": "How many malaria confirmed cases in Kano State in 2024?"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "aggregation"
    assert body["n_supporting_values"] > 0
    assert body["triplets"]
    assert body["record_ids"]
    assert body["latency_s"] < 10  # non-functional requirement S3.2.2


def test_query_ambiguous_flagged(client):
    r = client.post("/query", json={"question": "Where is the current outbreak?"})
    assert r.status_code == 200
    assert r.json()["intent"] == "ambiguous"


def test_baseline_endpoint(client):
    r = client.post("/baseline/query", json={
        "question": "How many malaria confirmed cases in Kano State in 2024?"})
    assert r.status_code == 200
    assert r.json()["mode"].startswith("baseline")


def test_query_validation(client):
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 422
