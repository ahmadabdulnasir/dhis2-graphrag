"""Knowledge graph schema (Figure 3.3 of the write-up).

Node labels:
    Indicator, Facility, LGA, State, Region, Period, DataValue

Relationships:
    (DataValue)-[:OF_INDICATOR]->(Indicator)
    (DataValue)-[:REPORTED_AT]->(Facility)
    (DataValue)-[:FOR_PERIOD]->(Period)
    (Facility)-[:LOCATED_IN]->(LGA)
    (LGA)-[:LOCATED_IN]->(State)
    (State)-[:PART_OF]->(Region)

A retrieved fact is expressed as a triplet such as
    [Malaria confirmed cases] -[OCCURRED_IN]-> [Kano State]
derived by traversing DataValue -> Facility -> LGA -> State.
"""

NODE_LABELS = ["Indicator", "Facility", "LGA", "State", "Region", "Period", "DataValue"]

REL_TYPES = ["OF_INDICATOR", "REPORTED_AT", "FOR_PERIOD", "LOCATED_IN", "PART_OF"]


def node_id(label: str, key: str) -> str:
    return f"{label.lower()}:{key}"
