from copy import deepcopy
from decimal import Decimal
import csv
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from banksim.data import (DataError, FIELDS, clean_category, load_validate_events,
                          parse_money, sha256_file, validate_records)
from banksim.panel import build_daily_panel
from banksim.quality import compare_reference
from banksim.visibility import visible_attributes


def transaction(**changes):
    record = dict(zip(FIELDS, (0, "C1", "2", "F", "28007", "M1", "28007", "es_food", "10.00", 0)))
    record.update(changes)
    return record


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.acceptance("D04", "D08", "D09", "D10")
def test_toy_fixture_expected_data():
    toy = json.loads(Path("examples/toy_walkthrough.json").read_text(encoding="utf-8"))
    events = validate_records(toy["raw_events"], tuple(toy["coverage"]))
    panel, registry = build_daily_panel(events, tuple(toy["coverage"]))
    expected = toy["expected_data"]
    assert len(events.raw_events) == expected["raw_rows"]
    assert len(events.normal_events) == expected["normal_rows"]
    assert events.normal_events["amount_cents"].sum() == expected["normal_amount_cents"]
    assert panel.loc[panel.customer.eq("T"), "target_cents"].tolist() == expected["customer_T_daily_cents"]
    assert len(panel) == 3 * 6
    assert panel.observed.all()
    assert registry.first_seen_step.tolist() == [0, 0, 0]


@pytest.mark.acceptance("D03")
def test_outer_quotes_only_and_string_ids(tmp_path):
    row = transaction(customer="  'C1093826151'  ", merchant=" 'M'INNER' ", age=" '2' ",
                      gender=" 'F' ", zipcodeOri=" '28007' ", zipMerchant=" '28007' ", category=" 'es_food' ")
    path = tmp_path / "quoted.csv"
    write_csv(path, [row])
    result = load_validate_events(path).raw_events.iloc[0]
    assert result.customer == "C1093826151"
    assert result.merchant == "M'INNER"
    assert result.zipcodeOri == result.zipMerchant == "28007"
    assert isinstance(result.customer, str)
    assert clean_category(" ''inside'' ") == "'inside'"


@pytest.mark.acceptance("D04")
@pytest.mark.parametrize("text,cents", [("0", 0), ("0.10", 10), ("1.23", 123), ("1234.56", 123456), ("0.29", 29)])
def test_decimal_cents(text, cents):
    amount, actual = parse_money(text)
    assert isinstance(amount, Decimal)
    assert actual == cents


@pytest.mark.acceptance("D05")
@pytest.mark.parametrize("field,value", [
    ("amount", "1.234"), ("amount", "1.230"), ("amount", "-1"), ("amount", "NaN"),
    ("amount", "Infinity"), ("amount", "-Infinity"), ("amount", ""), ("amount", 0.1),
    ("age", "7"), ("age", ""), ("gender", "X"), ("category", "es_nonexistent"),
    ("fraud", 2), ("fraud", "0.0"), ("step", -1), ("step", 180), ("step", "1.5"),
    ("customer", "''"), ("customer", "  "), ("merchant", ""), ("zipMerchant", "99999"),
])
def test_invalid_values_block_whole_input(field, value):
    with pytest.raises(DataError) as caught:
        validate_records([transaction(), transaction(**{field: value})])
    assert caught.value.code == "INVALID_VALUE"
    assert caught.value.details["validation_errors"][0]["raw_row_id"] == 2
    assert caught.value.details["rows_read"] == 2


@pytest.mark.acceptance("D05")
def test_missing_schema_and_malformed_rows(tmp_path):
    row = transaction()
    del row["age"]
    with pytest.raises(DataError, match="no partial") as caught:
        validate_records([row])
    assert caught.value.code == "INVALID_SCHEMA"
    for index, content in enumerate((
        "step,customer\n0,C1\n", ",".join(FIELDS) + "\n0,C1\n",
        ",".join(FIELDS) + "\n\n", ",".join(FIELDS) + '\n"unterminated',
        ",".join(FIELDS[:-1] + ("step",)) + "\n", "",
    )):
        path = tmp_path / f"bad{index}.csv"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(DataError) as error:
            load_validate_events(path)
        assert error.value.code == "INVALID_SCHEMA"


@pytest.mark.acceptance("D05")
def test_cli_nonzero_error_and_no_partial_parquets(tmp_path):
    path = tmp_path / "invalid.csv"
    write_csv(path, [transaction(amount="1.234")])
    output = tmp_path / "failed_output"
    result = subprocess.run([sys.executable, "src/banksim/data_stage.py", "--mode", "full",
                             "--raw", path.relative_to(Path.cwd()).as_posix(),
                             "--output", output.relative_to(Path.cwd()).as_posix()],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 2, result.stderr
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "INVALID_VALUE"
    assert report["source"]["sha256_before"] == report["source"]["sha256_after"]
    assert not list(output.glob("*.parquet"))


@pytest.mark.acceptance("D06")
def test_exact_duplicates_reported_and_retained():
    row = transaction()
    events = validate_records([row, deepcopy(row), transaction(amount="20")])
    assert len(events.raw_events) == 3
    assert events.diagnostics["exact_duplicate_rows"] == 1
    assert events.diagnostics["duplicates"] == [{"first_raw_row_id": 1, "duplicate_raw_row_id": 2}]
    panel, _ = build_daily_panel(events)
    assert panel.iloc[0].target_cents == 4000
    assert panel.iloc[0].raw_tx_count == 3


@pytest.mark.acceptance("D07")
@pytest.mark.parametrize("changes", [{"age": "3"}, {"gender": "M"}, {"age": "3", "gender": "M"}])
def test_attribute_conflicts_keep_sources_and_block_customer(changes):
    events = validate_records([transaction(), transaction(step=1, **changes), transaction(customer="C2")])
    assert len(events.raw_events) == 3
    conflict = events.diagnostics["attribute_conflicts"][0]
    assert conflict["code"] == "ATTRIBUTE_CONFLICT"
    assert conflict["customer"] == "C1"
    assert [row["raw_row_id"] for row in conflict["sources"]] == [1, 2]
    _, registry = build_daily_panel(events)
    blocked = registry.set_index("customer").loc["C1"]
    assert blocked.attribute_status == "ATTRIBUTE_CONFLICT"
    assert pd.isna(blocked.age) and pd.isna(blocked.gender)
    with pytest.raises(DataError) as caught:
        visible_attributes(events, "C1", 0, 1)
    assert caught.value.code == "ATTRIBUTE_CONFLICT"
    assert visible_attributes(events, "C2", 0, 1)["age"] == "2"


@pytest.mark.acceptance("D07")
def test_cli_attribute_conflict_has_nonzero_exit_and_source_evidence(tmp_path):
    path = tmp_path / "conflict.csv"
    write_csv(path, [transaction(), transaction(step=1, age="3")])
    output = tmp_path / "conflict_output"
    result = subprocess.run([sys.executable, "src/banksim/data_stage.py", "--mode", "full",
                             "--raw", path.relative_to(Path.cwd()).as_posix(),
                             "--output", output.relative_to(Path.cwd()).as_posix()],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2, result.stderr
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert report["error"]["code"] == "ATTRIBUTE_CONFLICT"
    assert [item["raw_row_id"] for item in report["error"]["conflicts"][0]["sources"]] == [1, 2]
    assert not list(output.glob("*.parquet"))


@pytest.mark.acceptance("D12", "D13", "D14")
def test_unsorted_raw_attribute_provenance_uses_earliest_allowed_record():
    events = validate_records([transaction(step=5), transaction(step=2, fraud=1), transaction(step=2)])
    _, registry = build_daily_panel(events)
    row = registry.iloc[0]
    assert row.first_seen_step == row.attribute_source_step == 2
    assert row.attribute_source_raw_row_id == 2
    attrs = visible_attributes(events, "C1", 3, 5)
    assert attrs["attribute_source_step"] == 5
    assert attrs["attribute_source_raw_row_id"] == 1


@pytest.mark.acceptance("D08")
def test_fraud_defines_start_but_not_amount_and_views_are_isolated():
    events = validate_records([transaction(fraud=1, amount="50"), transaction(step=3)])
    panel, registry = build_daily_panel(events)
    assert registry.first_seen_step.tolist() == [0]
    assert panel.head(4).target_cents.tolist() == [0, 0, 0, 1000]
    assert panel.head(4).observed.all()
    raw_copy = events.raw_events
    raw_copy.loc[0, "amount_cents"] = 1
    normal_copy = events.normal_events
    normal_copy.loc[:, "amount_cents"] = 2
    assert events.raw_events.amount_cents.tolist() == [5000, 1000]
    assert events.normal_events.amount_cents.tolist() == [1000]


@pytest.mark.acceptance("D10", "D12")
def test_before_first_after_last_and_fraud_only_customer():
    events = validate_records([transaction(step=2), transaction(customer="FRAUD_ONLY", step=4, fraud=1)])
    panel, registry = build_daily_panel(events)
    customer = panel.loc[panel.customer.eq("C1")].set_index("step")
    assert customer.loc[:1, "target_cents"].isna().all()
    assert not customer.loc[:1, "observed"].any()
    assert customer.loc[3:, "observed"].all()
    assert customer.loc[3:, "target_cents"].eq(0).all()
    fraud = panel.loc[panel.customer.eq("FRAUD_ONLY")].set_index("step")
    assert fraud.loc[4:, "target_cents"].eq(0).all()
    assert len(registry) == 2


def test_reference_mismatch_cannot_be_reported_as_pass():
    comparison = compare_reference({"raw": {"rows": 7}}, {"raw": {"rows": 8}})
    assert comparison == [{"name": "raw.rows", "actual": 7, "expected": 8, "matches": False}]


@pytest.mark.full
@pytest.mark.acceptance("D01")
def test_full_hash_before_after_and_manifest(full_outputs):
    report, audit = full_outputs["report"], full_outputs["audit"]
    actual = sha256_file(audit["source"]["path"])
    manifest = json.loads(Path("manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == audit["source"]["path"])
    assert actual == entry["sha256"] == audit["source"]["sha256"]
    assert report["source"]["sha256_before"] == report["source"]["sha256_after"] == actual


@pytest.mark.full
@pytest.mark.acceptance("D02")
def test_full_source_shape_types_and_row_ids(full_outputs):
    events, audit = full_outputs["events"], full_outputs["audit"]
    assert len(events) == audit["raw"]["rows"]
    assert events.customer.nunique() == audit["raw"]["customers"]
    assert set(events.columns) == set(FIELDS) | {"raw_row_id", "amount_cents"}
    assert full_outputs["report"]["diagnostics"]["source_columns"] == list(FIELDS)
    assert events.raw_row_id.tolist() == list(range(1, len(events) + 1))
    assert (events.step.min(), events.step.max(), events.step.nunique()) == (0, 179, 180)


@pytest.mark.full
@pytest.mark.acceptance("D09")
def test_full_normal_raw_fraud_amounts_and_zero_transactions(full_outputs):
    events, panel, audit = (full_outputs[key] for key in ("events", "panel", "audit"))
    normal, fraud = events.loc[events.fraud.eq(0)], events.loc[events.fraud.eq(1)]
    assert len(normal) == audit["normal_events"]["rows"]
    assert len(fraud) == audit["fraud_events"]["rows"]
    assert normal.customer.nunique() == audit["normal_events"]["customers"]
    assert normal.amount_cents.sum() == panel.target_cents.sum() == audit["normal_events"]["amount_cents_total"]
    assert fraud.amount_cents.sum() == audit["fraud_events"]["amount_cents_total"]
    assert events.amount_cents.sum() == audit["raw"]["amount_cents_total"]
    assert normal.amount_cents.eq(0).sum() == audit["normal_events"]["zero_amount_rows"]
    assert (panel.normal_tx_count.gt(0) & panel.target_cents.eq(0)).sum() == audit["daily_panel"]["normal_record_present_but_target_zero_days"]


@pytest.mark.full
@pytest.mark.acceptance("D10")
def test_full_grid_masks_and_integer_parquet(full_outputs):
    panel, audit = full_outputs["panel"], full_outputs["audit"]["daily_panel"]
    assert len(panel) == audit["rows"]
    assert not panel.duplicated(["customer", "step"]).any()
    assert panel.groupby("customer").size().eq(180).all()
    assert panel.observed.sum() == audit["observed_days"]
    assert (~panel.observed).sum() == panel.target_cents.isna().sum() == audit["unobserved_days"]
    assert (panel.observed & panel.target_cents.eq(0)).sum() == audit["observed_zero_target_days"]
    assert panel.target_cents.isna().equals(~panel.observed)
    schema = pq.read_schema("artifacts/data/daily_panel.parquet")
    assert schema.field("target_cents").type == pa.int64()
    assert schema.metadata[b"spec_version"] == b"1.0.0"
    assert schema.metadata[b"data_hash"].decode() == full_outputs["audit"]["source"]["sha256"]
    assert pq.read_schema("artifacts/data/events_clean.parquet").field("amount").type == pa.decimal128(20, 2)


@pytest.mark.full
@pytest.mark.acceptance("D11")
def test_full_first_seen_delays(full_outputs):
    events, registry = full_outputs["events"], full_outputs["registry"]
    first_raw = events.groupby("customer").step.min()
    first_normal = events.loc[events.fraud.eq(0)].groupby("customer").step.min()
    delays = first_normal - first_raw
    reference = full_outputs["audit"]["visibility_comparison"]
    assert delays.gt(0).sum() == reference["normal_start_later_than_raw_start_customers"]
    assert delays.max() == reference["maximum_delay_days"]
    assert registry.set_index("customer").first_seen_step.sort_index().equals(first_raw.sort_index())


@pytest.mark.full
def test_full_report_all_comparisons_and_persisted_metadata(full_outputs):
    report = full_outputs["report"]
    assert all(report["invariants"].values())
    assert all(item["matches"] for item in report["reference_comparisons"])
    for artifact in report["artifacts"]:
        assert sha256_file(artifact["path"]) == artifact["sha256"]
        assert artifact["read_back_equal"]
