"""DHIS2 Web API client (Track A extraction).

Pulls aggregate data values through the analytics endpoint and maps them to
the canonical RECORD_COLUMNS schema. Works against any DHIS2 2.38+ instance
(demo: https://play.dhis2.org/dev; later: NHMIS with proper access).

Auth: basic auth (username/password) or a personal access token.
Handles paging on metadata endpoints and retries with backoff on 429/5xx.
"""
from __future__ import annotations

import os
import time
import logging

import pandas as pd
import requests

from src.models import RECORD_COLUMNS, REGION_OF_STATE

log = logging.getLogger(__name__)


class DHIS2Client:
    def __init__(self, base_url: str | None = None, username: str | None = None,
                 password: str | None = None, pat: str | None = None,
                 timeout: int = 60, max_retries: int = 3):
        self.base_url = (base_url or os.getenv("DHIS2_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("DHIS2_BASE_URL not set")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        pat = pat or os.getenv("DHIS2_PAT")
        if pat:
            self.session.headers["Authorization"] = f"ApiToken {pat}"
        else:
            self.session.auth = (
                username or os.getenv("DHIS2_USERNAME", "admin"),
                password or os.getenv("DHIS2_PASSWORD", "district"),
            )

    # ---------- low level ----------
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/api/{path.lstrip('/')}"
        for attempt in range(self.max_retries):
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                log.warning("DHIS2 %s -> %s, retrying in %ss", path, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def _get_paged(self, path: str, key: str, params: dict | None = None) -> list[dict]:
        params = dict(params or {})
        params.setdefault("pageSize", 200)
        page, out = 1, []
        while True:
            params["page"] = page
            data = self._get(path, params)
            out.extend(data.get(key, []))
            pager = data.get("pager", {})
            if page >= pager.get("pageCount", 1):
                return out
            page += 1

    # ---------- metadata ----------
    def system_info(self) -> dict:
        return self._get("system/info.json")

    def org_units(self, level: int = 4) -> list[dict]:
        """Facilities with their ancestor chain (for LGA/state mapping)."""
        return self._get_paged(
            "organisationUnits.json", "organisationUnits",
            {"fields": "id,displayName,level,ancestors[id,displayName,level]",
             "filter": f"level:eq:{level}"},
        )

    def data_elements(self, name_filters: list[str] | None = None) -> list[dict]:
        params = {"fields": "id,displayName"}
        elements = self._get_paged("dataElements.json", "dataElements", params)
        if name_filters:
            lowered = [f.lower() for f in name_filters]
            elements = [e for e in elements
                        if any(f in e["displayName"].lower() for f in lowered)]
        return elements

    # ---------- data ----------
    def analytics(self, data_element_ids: list[str], org_unit_ids: list[str],
                  periods: list[str]) -> dict:
        """Aggregate values via /api/analytics (dx, ou, pe dimensions)."""
        params = {
            "dimension": [
                "dx:" + ";".join(data_element_ids),
                "ou:" + ";".join(org_unit_ids),
                "pe:" + ";".join(p.replace("-", "") for p in periods),
            ],
            "displayProperty": "NAME",
            "skipMeta": "false",
        }
        return self._get("analytics.json", params)

    def extract(self, data_element_ids: list[str], org_unit_ids: list[str],
                periods: list[str]) -> pd.DataFrame:
        """Run analytics and map the response to the canonical schema."""
        payload = self.analytics(data_element_ids, org_unit_ids, periods)
        return parse_analytics(payload)


def parse_analytics(payload: dict) -> pd.DataFrame:
    """Map a DHIS2 analytics response to RECORD_COLUMNS.

    Kept as a pure function so it can be unit-tested with fixtures without
    network access.
    """
    headers = [h["name"] for h in payload.get("headers", [])]
    items = payload.get("metaData", {}).get("items", {})
    idx = {name: headers.index(name) for name in ("dx", "ou", "pe", "value")}

    rows = []
    for i, row in enumerate(payload.get("rows", []), 1):
        dx, ou, pe = row[idx["dx"]], row[idx["ou"]], row[idx["pe"]]
        ou_meta = items.get(ou, {})
        ou_name = ou_meta.get("name", ou)
        # hierarchy is resolved separately via org_units(); default unknowns
        state = ou_meta.get("state", "Unknown")
        rows.append({
            "record_id": f"dhis2-{i:06d}",
            "indicator": items.get(dx, {}).get("name", dx),
            "indicator_id": dx,
            "facility": ou_name,
            "facility_id": ou,
            "lga": ou_meta.get("lga", "Unknown"),
            "state": state,
            "region": REGION_OF_STATE.get(state, "Unknown"),
            "period": f"{pe[:4]}-{pe[4:6]}" if len(pe) == 6 else pe,
            "value": float(row[idx["value"]]) if row[idx["value"]] not in ("", None) else float("nan"),
            "source": "dhis2",
        })
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)
