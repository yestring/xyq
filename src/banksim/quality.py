"""Compute observations first, then compare them with the reference audit."""

from __future__ import annotations

import pandas as pd

from .data import EventViews


def compute_statistics(events: EventViews, panel: pd.DataFrame, registry: pd.DataFrame) -> dict:
    raw, normal = events.raw_events, events.normal_events
    fraud = raw.loc[raw["fraud"].eq(1)]
    first_raw = raw.groupby("customer")["step"].min()
    first_normal = normal.groupby("customer")["step"].min()
    delays = first_normal - first_raw.reindex(first_normal.index)
    observed, target = panel["observed"], panel["target_cents"]
    conflicts = events.diagnostics["attribute_conflicts"]

    def customers_by(frame, field):
        return frame.drop_duplicates("customer")[field].value_counts().sort_index().to_dict()

    return {
        "raw": {
            "rows": len(raw), "customers": raw["customer"].nunique(),
            "merchants": raw["merchant"].nunique(), "categories": raw["category"].nunique(),
            "amount_cents_total": int(raw["amount_cents"].sum()),
            "missing_values": int(raw[list(raw.columns)].isna().sum().sum()),
            "exact_duplicate_rows": events.diagnostics["exact_duplicate_rows"],
            "customer_age_conflicts": sum("age" in c["fields"] for c in conflicts),
            "customer_gender_conflicts": sum("gender" in c["fields"] for c in conflicts),
            "zipcodeOri_unique_values": sorted(raw["zipcodeOri"].unique().tolist()),
            "zipMerchant_unique_values": sorted(raw["zipMerchant"].unique().tolist()),
            "customer_gender_counts": customers_by(raw, "gender"),
            "customer_age_counts": customers_by(raw, "age"),
        },
        "normal_events": {
            "rows": len(normal), "customers": normal["customer"].nunique(),
            "amount_cents_total": int(normal["amount_cents"].sum()),
            "zero_amount_rows": int(normal["amount_cents"].eq(0).sum()),
            "negative_amount_rows": int(normal["amount_cents"].lt(0).sum()),
            "age_unknown_customers": normal.loc[normal["age"].eq("U"), "customer"].nunique(),
            "age_unknown_rows": int(normal["age"].eq("U").sum()),
            "customer_gender_counts": customers_by(normal, "gender"),
            "customer_age_counts": customers_by(normal, "age"),
        },
        "fraud_events": {
            "rows": len(fraud), "amount_cents_total": int(fraud["amount_cents"].sum()),
            "only_fraud_customers": len(set(raw["customer"]) - set(normal["customer"])),
        },
        "time": {
            "step_min": int(raw["step"].min()), "step_max": int(raw["step"].max()),
            "distinct_steps": raw["step"].nunique(),
            "first_raw_step_min": int(first_raw.min()), "first_raw_step_max": int(first_raw.max()),
            "first_normal_step_min": int(first_normal.min()) if len(first_normal) else None,
            "first_normal_step_max": int(first_normal.max()) if len(first_normal) else None,
        },
        "visibility_comparison": {
            "normal_start_later_than_raw_start_customers": int(delays.gt(0).sum()),
            "maximum_delay_days": int(delays.max()) if len(delays) else None,
        },
        "daily_panel": {
            "rows": len(panel), "observed_days": int(observed.sum()),
            "unobserved_days": int((~observed).sum()),
            "observed_zero_target_days": int((observed & target.eq(0)).sum()),
            "observed_positive_target_days": int((observed & target.gt(0)).sum()),
            "target_cents_null_days": int(target.isna().sum()),
            "target_cents_total_ignoring_nulls": int(target.sum()),
            "fixed_panel_zero_count_if_nulls_temporarily_filled_with_zero_for_audit_only": int(target.fillna(0).eq(0).sum()),
            "normal_record_present_but_target_zero_days": int((panel["normal_tx_count"].gt(0) & target.eq(0)).sum()),
            "normal_record_absent_cells_in_full_fixed_grid": int(panel["normal_tx_count"].eq(0).sum()),
        },
        "registry": {"rows": len(registry)},
    }


def compare_reference(actual: dict, reference: dict) -> list[dict]:
    checks = []
    for section, values in actual.items():
        for key, value in values.items():
            if section in reference and key in reference[section]:
                expected = reference[section][key]
                checks.append({"name": f"{section}.{key}", "actual": value,
                               "expected": expected, "matches": value == expected})
    return checks


def data_invariants(events: EventViews, panel: pd.DataFrame, registry: pd.DataFrame, coverage) -> dict:
    raw, normal = events.raw_events, events.normal_events
    first = raw.groupby("customer")["step"].min()
    target = panel["target_cents"]
    return {
        "complete_unique_grid": len(panel) == len(first) * (coverage[1] - coverage[0] + 1)
            and not panel.duplicated(["customer", "step"]).any(),
        "raw_row_ids_preserved": raw["raw_row_id"].tolist() == list(range(1, len(raw) + 1)),
        "normal_amount_conserved": int(target.sum()) == int(normal["amount_cents"].sum()),
        "raw_tx_count_conserved": int(panel["raw_tx_count"].sum()) == len(raw),
        "normal_tx_count_conserved": int(panel["normal_tx_count"].sum()) == len(normal),
        "observed_matches_raw_start": bool(panel["observed"].eq(panel["step"].ge(panel["customer"].map(first))).all()),
        "null_iff_unobserved": target.isna().equals(~panel["observed"]),
        "targets_nonnegative": bool(target.dropna().ge(0).all()),
        "zero_when_no_normal_records": bool(target.loc[panel["observed"] & panel["normal_tx_count"].eq(0)].eq(0).all()),
        "registry_matches_raw_start": registry.set_index("customer")["first_seen_step"].sort_index().equals(first.sort_index()),
    }
