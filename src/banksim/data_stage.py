"""Run from the project root: python src/banksim/data_stage.py --mode toy|full."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from banksim import SPEC_VERSION
from banksim.data import DataError, FIELDS, load_validate_events, sha256_file, validate_records
from banksim.panel import build_daily_panel
from banksim.quality import compare_reference, compute_statistics, data_invariants


def json_default(value):
    if value is pd.NA:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def dump_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        raise DataError("INVALID_VALUE", "Paths must stay inside the project", path=str(path)) from exc


def environment() -> dict:
    return {
        "python": platform.python_version(), "executable": project_relative(Path(sys.executable)),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "pytest")},
    }


def write_parquet(frame: pd.DataFrame, path: Path, data_hash: str) -> dict:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if "amount" in frame:
        position = table.schema.get_field_index("amount")
        table = table.set_column(position, "amount", table.column("amount").cast(pa.decimal128(20, 2)))
    metadata = dict(table.schema.metadata or {})
    metadata.update(spec_version=SPEC_VERSION, data_hash=data_hash)
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="snappy")
    restored = pq.read_table(temporary)
    if not table.equals(restored, check_metadata=True):
        raise DataError("INVALID_VALUE", "Parquet read-back differs", artifact=project_relative(path))
    temporary.replace(path)
    return {"path": project_relative(path), "rows": len(frame), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "read_back_equal": True,
            "schema": {field.name: str(field.type) for field in restored.schema}}


def run(args, report: dict, output: Path):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected_data = {"coverage_start": 0, "coverage_end": 179, "normal_fraud_value": 0,
                     "money_scale": 100, "visibility_source": "raw_events"}
    if config.get("spec_version") != SPEC_VERSION or any(config.get("data", {}).get(k) != v for k, v in expected_data.items()):
        raise DataError("CONFIG_MISMATCH", "Stage-one config differs from specification 1.0.0")
    source = Path(args.toy if args.mode == "toy" else args.raw or config["data"]["raw_path"])
    project_relative(source)
    before = sha256_file(source)
    report["source"] = {"path": project_relative(source), "sha256_before": before, "bytes": source.stat().st_size}
    if args.mode == "toy":
        toy = json.loads(source.read_text(encoding="utf-8"))
        coverage = tuple(toy["coverage"])
        events = validate_records(toy["raw_events"], coverage)
    else:
        coverage = (config["data"]["coverage_start"], config["data"]["coverage_end"])
        events = load_validate_events(source, coverage)
    panel, registry = build_daily_panel(events, coverage)
    report.update(coverage=list(coverage), data_hash=before, diagnostics=events.diagnostics,
                  actual=compute_statistics(events, panel, registry),
                  invariants=data_invariants(events, panel, registry, coverage))
    checks = []
    if args.mode == "toy":
        expected = toy["expected_data"]
        actual = {"raw_rows": report["actual"]["raw"]["rows"],
                  "normal_rows": report["actual"]["normal_events"]["rows"],
                  "normal_amount_cents": report["actual"]["normal_events"]["amount_cents_total"],
                  "customer_T_daily_cents": panel.loc[panel["customer"].eq("T"), "target_cents"].tolist()}
        checks = [{"name": f"toy.{key}", "actual": actual[key], "expected": value,
                   "matches": actual[key] == value} for key, value in expected.items()]
        report["fixture_kind"] = toy["kind"]
    else:
        reference = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        if reference["spec_version"] != SPEC_VERSION:
            raise DataError("CONFIG_MISMATCH", "Audit specification version differs")
        checks = compare_reference(report["actual"], reference)
        for name, actual, expected in (
            ("source.sha256", before, reference["source"]["sha256"]),
            ("source.bytes", source.stat().st_size, reference["source"]["bytes"]),
            ("source.column_order", events.diagnostics["source_columns"], reference["source"]["column_order"]),
        ):
            checks.append({"name": name, "actual": actual, "expected": expected, "matches": actual == expected})
        report["reference_path"] = project_relative(Path(args.audit))
        report["reference_sha256"] = sha256_file(args.audit)
    after = sha256_file(source)
    report["source"]["sha256_after"] = after
    report["invariants"]["source_unchanged"] = before == after
    report["reference_comparisons"] = checks
    conflicts = events.diagnostics["attribute_conflicts"]
    if conflicts:
        raise DataError("ATTRIBUTE_CONFLICT", "Customer profiles blocked; conflicting attributes not selected",
                        conflicts=conflicts)
    if not all(report["invariants"].values()) or not all(check["matches"] for check in checks):
        raise DataError("INVALID_VALUE", "Computed statistics or invariants failed; artifacts not published")
    report["artifacts"] = [write_parquet(frame, output / name, before) for name, frame in (
        ("events_clean.parquet", events.raw_events),
        ("customer_registry.parquet", registry),
        ("daily_panel.parquet", panel),
    )]
    report["examples"] = {
        "events_head": events.raw_events.head(2).to_dict("records"),
        "registry_head": registry.head(2).to_dict("records"),
        "panel_head": panel.head(3).to_dict("records"),
        "panel_tail": panel.tail(3).to_dict("records"),
        "unobserved": panel.loc[~panel["observed"]].head(2).to_dict("records"),
        "observed_zero": panel.loc[panel["observed"] & panel["target_cents"].eq(0)].head(2).to_dict("records"),
    }
    report["status"] = "PASS"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("toy", "full"))
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--raw")
    parser.add_argument("--toy", default="examples/toy_walkthrough.json")
    parser.add_argument("--audit", default="data/reference/normal_data_audit.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    output = Path(args.output or ("artifacts/smoke/data" if args.mode == "toy" else "artifacts/data"))
    artifacts_root = (Path.cwd() / "artifacts").resolve()
    if not output.resolve().is_relative_to(artifacts_root):
        print(json.dumps({"code": "INVALID_VALUE", "message": "Output must be inside artifacts/"}))
        return 2
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"spec_version": SPEC_VERSION, "stage": 1, "mode": "smoke" if args.mode == "toy" else "full",
              "status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat()}
    try:
        report["environment"] = environment()
        run(args, report, output)
    except DataError as exc:
        report.update(status="FAILED", error=exc.as_dict())
    except (OSError, KeyError, ValueError) as exc:
        report.update(status="FAILED", error={"code": "INVALID_SCHEMA", "message": str(exc)})
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if "source" in report:
        report["source"]["sha256_after"] = sha256_file(report["source"]["path"])
        if report["source"]["sha256_after"] != report["source"]["sha256_before"]:
            report.update(status="FAILED", error={"code": "INVALID_VALUE", "message": "Source changed during run"})
    dump_json(output / "quality_report.json", report)
    print(json.dumps({"status": report["status"], "mode": report["mode"],
                      "report": project_relative(output / "quality_report.json"),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "actual": report.get("actual"), "error": report.get("error")}, default=json_default))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
