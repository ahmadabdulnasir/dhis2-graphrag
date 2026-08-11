"""Nigeria Health Watch open-data loader.

Loads CSV exports downloaded from the portal and maps them to the canonical
schema. Column names differ between exports, so the loader takes a mapping;
sensible defaults cover the common layout (indicator, state, lga, facility,
period, value).
"""
from __future__ import annotations

import pandas as pd

from src.models import RECORD_COLUMNS, REGION_OF_STATE

DEFAULT_MAPPING = {
    "indicator": "indicator",
    "facility": "facility",
    "lga": "lga",
    "state": "state",
    "period": "period",
    "value": "value",
}


def load_csv(path: str, mapping: dict | None = None, source: str = "nhw") -> pd.DataFrame:
    mapping = {**DEFAULT_MAPPING, **(mapping or {})}
    raw = pd.read_csv(path)
    missing = [c for c in mapping.values() if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns {missing}. "
            f"Pass a mapping= dict, available columns: {list(raw.columns)}"
        )

    df = pd.DataFrame()
    df["indicator"] = raw[mapping["indicator"]].astype(str).str.strip()
    df["indicator_id"] = df["indicator"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    df["facility"] = raw[mapping["facility"]].astype(str).str.strip()
    df["facility_id"] = df["facility"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True)
    df["lga"] = raw[mapping["lga"]].astype(str).str.strip()
    df["state"] = raw[mapping["state"]].astype(str).str.strip()
    df["region"] = df["state"].map(REGION_OF_STATE).fillna("Unknown")
    df["period"] = _normalise_period(raw[mapping["period"]])
    df["value"] = pd.to_numeric(raw[mapping["value"]], errors="coerce")
    df["source"] = source
    df["record_id"] = [f"{source}-{i:06d}" for i in range(1, len(df) + 1)]
    return df[RECORD_COLUMNS]


def _normalise_period(s: pd.Series) -> pd.Series:
    """Accept '2024-03', '202403', '03/2024', 'Mar-2024' and return ISO month."""
    parsed = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    # fall back for DHIS2-style '202403'
    mask = parsed.isna()
    if mask.any():
        compact = s.astype(str).str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
        parsed = parsed.fillna(pd.to_datetime(compact, errors="coerce", format="mixed"))
    return parsed.dt.strftime("%Y-%m")
