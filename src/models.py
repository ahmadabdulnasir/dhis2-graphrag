"""Canonical internal data model.

Every extraction source (DHIS2 API, Nigeria Health Watch CSV, synthetic
generator) produces a pandas DataFrame with the RECORD_COLUMNS schema below.
Everything downstream (validation, graph construction, retrieval) works
only with this schema, so sources are interchangeable.
"""
from dataclasses import dataclass, field

# One row = one aggregate data value (DHIS2 aggregate model)
RECORD_COLUMNS = [
    "record_id",      # stable id, used for provenance
    "indicator",      # human-readable indicator name, e.g. "Malaria confirmed cases"
    "indicator_id",   # source id (DHIS2 dataElement uid or slug)
    "facility",       # reporting facility name
    "facility_id",    # source org unit uid or slug
    "lga",            # local government area
    "state",          # state
    "region",         # geopolitical region ("North" / "South" for now)
    "period",         # ISO month, e.g. "2024-03"
    "value",          # numeric value (float; NaN when missing)
    "source",         # "dhis2" | "nhw" | "synthetic"
]

INDICATORS = {
    "anc_1st_visit": "ANC 1st visit",
    "anc_4th_visit": "ANC 4th visit",
    "facility_delivery": "Facility deliveries",
    "malaria_confirmed": "Malaria confirmed cases",
    "tb_notified": "TB cases notified",
    "penta3": "Penta 3 doses given",
}

# Synonyms used by the entity linker when parsing questions
INDICATOR_SYNONYMS = {
    "anc_1st_visit": ["anc 1", "anc first", "first antenatal", "antenatal first", "anc 1st"],
    "anc_4th_visit": ["anc 4", "anc fourth", "fourth antenatal", "anc 4th"],
    "facility_delivery": ["deliveries", "delivery", "births", "facility delivery",
                          "maternal delivery"],
    "malaria_confirmed": ["malaria", "malaria cases", "malaria incidence"],
    "tb_notified": ["tb", "tuberculosis", "tb cases"],
    "penta3": ["penta", "penta3", "immunisation", "immunization", "vaccination",
               "immunisation coverage", "vaccine"],
}

REGION_OF_STATE = {
    "Kano": "North",
    "Kaduna": "North",
    "Borno": "North",
    "Lagos": "South",
    "Rivers": "South",
    "Enugu": "South",
}


@dataclass
class ValidationReport:
    """Result of the data preparation/validation stage (objective 1)."""
    n_records: int = 0
    completeness: float = 0.0        # target >= 0.95
    consistency: float = 0.0         # target >= 0.90
    n_missing: int = 0
    n_outliers: int = 0
    n_logical_errors: int = 0
    n_dropped: int = 0
    notes: list = field(default_factory=list)

    @property
    def meets_targets(self) -> bool:
        return bool(self.completeness >= 0.95 and self.consistency >= 0.90)

    def summary(self) -> str:
        return (
            f"records={self.n_records}, completeness={self.completeness:.1%} "
            f"(target 95%), consistency={self.consistency:.1%} (target 90%), "
            f"missing={self.n_missing}, outliers={self.n_outliers}, "
            f"logical_errors={self.n_logical_errors}, dropped={self.n_dropped}"
        )
