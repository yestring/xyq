"""Access only the allowed history; registry attributes cannot restore hidden data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import DataError, EventViews, attribute_conflicts


def visible_attributes(events: EventViews, customer: str, history_start: int, origin_t: int) -> dict:
    if history_start > origin_t:
        raise DataError("INVALID_VALUE", "History start is after origin")
    raw = events.raw_events
    rows = raw.loc[raw["customer"].eq(customer) & raw["step"].between(history_start, origin_t)]
    conflicts = attribute_conflicts(rows)
    if conflicts:
        raise DataError("ATTRIBUTE_CONFLICT", "Visible customer attributes conflict", conflicts=conflicts)
    if rows.empty:
        return {"age": "U", "gender": "U", "attribute_source_step": None,
                "attribute_source_raw_row_id": None, "attribute_status": "UNKNOWN"}
    source = rows.sort_values(["step", "raw_row_id"], kind="stable").iloc[0]
    return {"age": source["age"], "gender": source["gender"],
            "attribute_source_step": int(source["step"]),
            "attribute_source_raw_row_id": int(source["raw_row_id"]), "attribute_status": "VISIBLE"}


def require_continuous_observed(panel: pd.DataFrame, customer: str, start: int, end: int) -> pd.DataFrame:
    if start > end:
        raise DataError("INVALID_VALUE", "Window start is after end")
    rows = panel.loc[panel["customer"].eq(customer) & panel["step"].between(start, end)].sort_values("step")
    if rows["step"].tolist() != list(range(start, end + 1)):
        raise DataError("DATA_GAP", "Window has missing or duplicate days", customer=customer, start=start, end=end)
    if not rows["observed"].eq(True).all() or rows["target_cents"].isna().any():
        raise DataError("DATA_GAP", "Window contains unobserved or missing targets", customer=customer,
                        start=start, end=end)
    return rows.copy()


@dataclass(frozen=True)
class VisibleWindow:
    customer: str
    origin_t: int
    history_start: int
    Tobs: int
    history: pd.DataFrame
    attributes: dict
    labels: pd.DataFrame | None


def read_visible_window(
    events: EventViews, panel: pd.DataFrame, customer: str, origin_t: int,
    *, Lc: int | None = None, H: int = 7, require_labels: bool = True,
) -> VisibleWindow:
    """A data contract, not an experiment splitter or prediction implementation.

    Tobs counts every continuous visible day, including zero-consumption days.
    A real inference request sets require_labels=False and gets no future data.
    """
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (origin_t, H)) or H <= 0:
        raise DataError("INVALID_VALUE", "Origin and positive horizon must be integers")
    if Lc is not None and (isinstance(Lc, bool) or not isinstance(Lc, int) or Lc <= 0):
        raise DataError("INVALID_VALUE", "Lc must be a positive integer")
    raw = events.raw_events
    known = raw.loc[raw["customer"].eq(customer) & raw["step"].le(origin_t)]
    if known.empty:
        raise DataError("NO_HISTORY", "Customer is not yet visible", customer=customer, origin_t=origin_t)
    first_seen = int(known["step"].min())
    start = first_seen if Lc is None else origin_t - Lc + 1
    if start < first_seen:
        raise DataError("DATA_GAP", "Customer does not have exactly Lc visible days", first_seen_step=first_seen)
    history = require_continuous_observed(panel, customer, start, origin_t)
    attrs = visible_attributes(events, customer, start, origin_t)
    labels = None
    if require_labels:
        labels = require_continuous_observed(panel, customer, origin_t + 1, origin_t + H)
    return VisibleWindow(customer, origin_t, start, len(history), history, attrs, labels)
