"""Synthetic DHIS2-style aggregate dataset (Track B benchmark).

Generates monthly facility-level values for six indicators across six
Nigerian states (three north, three south), 2020-2025, with realistic
structure the evaluation queries depend on:

- malaria peaks in the rainy season (Jun-Sep), higher burden in the north
- immunisation (penta3) trends upward in Lagos over the years
- ANC4 is always <= ANC1 in truth; violations are injected as errors
- ~3% missing values and ~1% outliers injected so validation has work to do

Seed is fixed (42) so ground-truth answers computed from this data are stable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import INDICATORS, RECORD_COLUMNS, REGION_OF_STATE

SEED = 42

LGAS = {
    "Kano": ["Kano Municipal", "Nassarawa", "Dala"],
    "Kaduna": ["Kaduna North", "Zaria", "Chikun"],
    "Borno": ["Maiduguri", "Jere", "Konduga"],
    "Lagos": ["Ikeja", "Surulere", "Alimosho"],
    "Rivers": ["Port Harcourt", "Obio-Akpor", "Eleme"],
    "Enugu": ["Enugu North", "Nsukka", "Udi"],
}

# baseline monthly value per facility for each indicator
BASE_LEVEL = {
    "anc_1st_visit": 55,
    "anc_4th_visit": 38,
    "facility_delivery": 32,
    "malaria_confirmed": 120,
    "tb_notified": 6,
    "penta3": 70,
}

MISSING_RATE = 0.03
OUTLIER_RATE = 0.010
LOGICAL_ERROR_RATE = 0.008


def _months(start="2020-01", end="2025-12") -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def generate(start="2020-01", end="2025-12") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = _months(start, end)
    rows = []
    rid = 0

    for state, lgas in LGAS.items():
        region = REGION_OF_STATE[state]
        north = region == "North"
        for lga in lgas:
            for f_idx in range(3):
                facility = f"{lga} PHC {f_idx + 1}"
                facility_id = f"{state[:2].upper()}-{lga.replace(' ', '')[:6].upper()}-{f_idx + 1}"
                # facility size factor, stable per facility
                size = rng.uniform(0.6, 1.6)
                anc1_prev = None
                for period in months:
                    year = int(period[:4])
                    month = int(period[5:])
                    for slug, name in INDICATORS.items():
                        base = BASE_LEVEL[slug] * size
                        if slug == "malaria_confirmed":
                            seasonal = 1.0 + (0.7 if month in (6, 7, 8, 9) else 0.0)
                            burden = 1.5 if north else 1.0
                            mean = base * seasonal * burden
                        elif slug == "penta3":
                            trend = 1.0 + (year - 2020) * (0.08 if state == "Lagos" else 0.02)
                            mean = base * trend
                        elif slug == "anc_4th_visit":
                            # tie to this facility/period's ANC1 so the logical
                            # rule ANC4 <= ANC1 holds in the true data
                            mean = (anc1_prev or base) * 0.7
                        else:
                            mean = base
                        value = float(max(0, rng.normal(mean, mean * 0.15)))
                        value = round(value)
                        if slug == "anc_1st_visit":
                            anc1_prev = value
                        if slug == "anc_4th_visit" and anc1_prev is not None:
                            value = min(value, anc1_prev)
                        rid += 1
                        rows.append({
                            "record_id": f"syn-{rid:06d}",
                            "indicator": name,
                            "indicator_id": slug,
                            "facility": facility,
                            "facility_id": facility_id,
                            "lga": lga,
                            "state": state,
                            "region": region,
                            "period": period,
                            "value": float(value),
                            "source": "synthetic",
                        })

    df = pd.DataFrame(rows, columns=RECORD_COLUMNS)

    # --- inject data-quality problems (indices deterministic via rng) ---
    n = len(df)
    missing_idx = rng.choice(n, size=int(n * MISSING_RATE), replace=False)
    df.loc[missing_idx, "value"] = np.nan

    remaining = df.index.difference(missing_idx)
    outlier_idx = rng.choice(remaining, size=int(n * OUTLIER_RATE), replace=False)
    df.loc[outlier_idx, "value"] = df.loc[outlier_idx, "value"] * rng.integers(8, 20, size=len(outlier_idx))

    # logical errors: ANC4 > ANC1 for a small share of facility-periods
    anc4 = df[(df.indicator_id == "anc_4th_visit") & df.value.notna()].index
    err_idx = rng.choice(anc4, size=int(len(anc4) * LOGICAL_ERROR_RATE * 6), replace=False)
    df.loc[err_idx, "value"] = df.loc[err_idx, "value"] * 5 + 500

    return df


if __name__ == "__main__":
    d = generate()
    print(d.shape)
    print(d.head())
