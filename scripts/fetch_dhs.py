"""Fetch real Nigerian health indicator data from the DHS Program API.

The DHS API is public (no key) and serves state-level (subnational) values
from the Nigeria Demographic and Health Surveys. Run from a machine with
open internet:

    uv run python scripts/fetch_dhs.py                 # fetch default indicators
    uv run python scripts/fetch_dhs.py --list malaria  # search indicator codes

Output: data/dhs_nigeria.csv in the pipeline's canonical column layout.
Load it with:

    uv run python run_pipeline.py --source data/dhs_nigeria.csv

Notes on mapping: DHS data is survey-based, so 'period' is the survey year
(e.g. '2018'), not a month; aggregation and comparison queries work the same
way since period matching is prefix-based. 'facility' is set to the state
name (DHS reports at state level, not facility level).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import REGION_OF_STATE  # noqa: E402

API = "https://api.dhsprogram.com/rest/dhs"

# Geopolitical-zone aggregate rows are excluded (keeping them alongside
# states would double-count national totals); the zone mapping below gives
# every state its region instead.
ZONES = {"north central", "north east", "north west",
         "south east", "south south", "south west", "nigeria", "total"}

NORTH_STATES = {
    "Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "FCT",
    "FCT Abuja", "Abuja", "Adamawa", "Bauchi", "Borno", "Gombe", "Taraba",
    "Yobe", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara",
}
SOUTH_STATES = {
    "Abia", "Anambra", "Ebonyi", "Enugu", "Imo", "Akwa Ibom", "Bayelsa",
    "Cross River", "Delta", "Edo", "Rivers", "Ekiti", "Lagos", "Ogun",
    "Ondo", "Osun", "Oyo",
}

# Standard DHS indicator codes relevant to the thesis focus areas
DEFAULT_INDICATORS = {
    "RH_ANCN_W_N4P": "ANC 4+ visits (%)",
    "RH_DELP_C_DHF": "Delivery in a health facility (%)",
    "CH_VACC_C_BAS": "Children with basic vaccinations (%)",
    "ML_PMAL_C_RDT": "Malaria prevalence in children (RDT) (%)",
    "ML_NETC_C_ITN": "Children who slept under an ITN (%)",
}


def list_indicators(term: str):
    r = requests.get(f"{API}/indicators",
                     params={"countryIds": "NG", "perPage": 5000, "f": "json"},
                     timeout=60)
    r.raise_for_status()
    for row in r.json()["Data"]:
        label = row.get("Label", "")
        if term.lower() in label.lower():
            print(f"{row['IndicatorId']:<22} {label}")


def fetch(indicators: dict[str, str], out_path: Path):
    rows_out = []
    rid = 0
    for code, name in indicators.items():
        page, fetched = 1, 0
        while True:
            r = requests.get(f"{API}/data", params={
                "countryIds": "NG", "indicatorIds": code,
                "breakdown": "subnational", "surveyYearStart": 2008,
                "perPage": 1000, "page": page, "f": "json"}, timeout=120)
            r.raise_for_status()
            payload = r.json()
            data = payload.get("Data", [])
            for d in data:
                # CharacteristicLabel holds the state name, with '..' padding
                state = str(d.get("CharacteristicLabel", "")).strip(". ")
                value = d.get("Value")
                year = d.get("SurveyYear")
                if not state or value is None:
                    continue
                if state.lower() in ZONES:
                    continue  # zone aggregates would double-count states
                rid += 1
                rows_out.append({
                    "record_id": f"dhs-{rid:06d}",
                    "indicator": name,
                    "indicator_id": code.lower(),
                    "facility": state,          # DHS reports at state level
                    "facility_id": f"dhs-{state.lower().replace(' ', '-')}",
                    "lga": state,               # no LGA breakdown in DHS
                    "state": state,
                    "region": ("North" if state in NORTH_STATES
                               else "South" if state in SOUTH_STATES
                               else REGION_OF_STATE.get(state, "Unknown")),
                    "period": str(year),
                    "value": float(value),
                    "source": "dhs",
                })
            fetched += len(data)
            total_pages = payload.get("TotalPages", 1)
            if page >= total_pages:
                break
            page += 1
        print(f"  {code}: {fetched} rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {len(rows_out)} records -> {out_path}")
    print("run: uv run python run_pipeline.py --source", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", metavar="TERM", help="search indicator codes by name")
    ap.add_argument("--out", default="data/dhs_nigeria.csv")
    args = ap.parse_args()
    if args.list:
        list_indicators(args.list)
    else:
        fetch(DEFAULT_INDICATORS, Path(args.out))
