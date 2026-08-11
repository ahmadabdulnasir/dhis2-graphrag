"""Data preparation and validation stage (pipeline stage 1, objective 1).

Bounded preprocessing modelled on the WHO Data Quality Review toolkit:

- completeness: share of expected facility-indicator-period cells that carry
  a value (target >= 95%)
- consistency: share of present values that pass outlier and logical checks
  (target >= 90%)

validate() returns (clean_df, ValidationReport). Records that fail checks are
excluded from the knowledge graph so answers are grounded in trusted values;
the report is kept for the thesis write-up (Chapter 5 data-quality table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import ValidationReport

Z_THRESHOLD = 3.5  # robust z-score on facility-indicator history


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport(n_records=len(df))

    # ---- completeness ----
    missing_mask = df["value"].isna()
    report.n_missing = int(missing_mask.sum())
    report.completeness = 1.0 - report.n_missing / max(len(df), 1)

    present = df[~missing_mask].copy()

    # ---- consistency: robust outlier check per facility-indicator series ----
    grp = present.groupby(["facility_id", "indicator_id"])["value"]
    median = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, np.nan)
    robust_z = 0.6745 * (present["value"] - median) / mad
    outlier_mask = robust_z.abs() > Z_THRESHOLD
    outlier_mask = outlier_mask.fillna(False) | (present["value"] < 0)
    report.n_outliers = int(outlier_mask.sum())

    # ---- consistency: logical rule ANC4 <= ANC1 per facility-period ----
    wide = present.pivot_table(index=["facility_id", "period"], columns="indicator_id",
                               values="value", aggfunc="first")
    logical_bad = set()
    if {"anc_1st_visit", "anc_4th_visit"} <= set(wide.columns):
        bad = wide[wide["anc_4th_visit"] > wide["anc_1st_visit"]].index
        logical_bad = set(bad)
    logical_mask = present.apply(
        lambda r: r["indicator_id"] == "anc_4th_visit"
        and (r["facility_id"], r["period"]) in logical_bad,
        axis=1,
    ) if logical_bad else pd.Series(False, index=present.index)
    report.n_logical_errors = int(logical_mask.sum())

    flagged = outlier_mask | logical_mask
    report.consistency = 1.0 - flagged.sum() / max(len(present), 1)

    clean = present[~flagged].copy()
    report.n_dropped = report.n_missing + int(flagged.sum())
    report.notes.append(
        "targets met" if report.meets_targets else "targets NOT met — review source data"
    )
    return clean, report
