"""Construct the raw-based visibility grid and normal-only daily targets."""

from __future__ import annotations

import pandas as pd

from .data import EventViews, validate_coverage


def build_daily_panel(events: EventViews, coverage=(0, 179)) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = validate_coverage(coverage)
    raw = events.raw_events
    normal = events.normal_events
    first = raw.sort_values(["customer", "step", "raw_row_id"], kind="stable").drop_duplicates("customer")
    registry = first[["customer", "step", "age", "gender", "raw_row_id"]].rename(columns={
        "step": "first_seen_step", "raw_row_id": "attribute_source_raw_row_id",
    }).reset_index(drop=True)
    registry["attribute_source_step"] = registry["first_seen_step"].astype("Int64")
    registry["attribute_source_raw_row_id"] = registry["attribute_source_raw_row_id"].astype("Int64")
    registry["attribute_status"] = "CONSISTENT"
    conflicted_ids = {item["customer"] for item in events.diagnostics["attribute_conflicts"]}
    bad = registry["customer"].isin(conflicted_ids)
    registry.loc[bad, ["age", "gender", "attribute_source_step", "attribute_source_raw_row_id"]] = pd.NA
    registry.loc[bad, "attribute_status"] = "ATTRIBUTE_CONFLICT"
    index = pd.MultiIndex.from_product(
        [registry["customer"].tolist(), range(start, end + 1)], names=["customer", "step"],
    )
    raw_count = raw.groupby(["customer", "step"]).size().reindex(index, fill_value=0)
    normal_count = normal.groupby(["customer", "step"]).size().reindex(index, fill_value=0)
    amounts = normal.groupby(["customer", "step"])["amount_cents"].sum().reindex(index, fill_value=0)
    panel = pd.DataFrame({
        "raw_tx_count": raw_count.astype("int64"),
        "normal_tx_count": normal_count.astype("int64"),
        "target_cents": amounts.astype("Int64"),
    }).reset_index()
    panel["customer"] = panel["customer"].astype("string")
    first_steps = registry.set_index("customer")["first_seen_step"]
    panel["observed"] = panel["step"].ge(panel["customer"].map(first_steps)).astype(bool)
    panel.loc[~panel["observed"], "target_cents"] = pd.NA
    panel = panel[["customer", "step", "observed", "target_cents", "raw_tx_count", "normal_tx_count"]]
    return panel, registry
