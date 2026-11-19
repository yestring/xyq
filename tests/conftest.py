"""Persist acceptance status from actual pytest outcomes, never preset PASS."""

import json
from pathlib import Path

import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption("--forecast-artifacts-root", default="artifacts/stage4",
                     help="Root containing newly generated smoke models; old artifacts stay immutable")
    parser.addoption("--acceptance-report", help="Write actual acceptance execution results inside artifacts/")
    parser.addoption("--acceptance-stage", type=int, choices=(1, 2, 3, 4), default=1,
                     help="1: data; 2: clustering; 3: retrieval; 4: forecast 4A scoped acceptance")


def pytest_configure(config):
    config._stage_one_results = {}
    config._stage_one_ids = {}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        config._stage_one_ids[item.nodeid] = [
            identifier for marker in item.iter_markers("acceptance") for identifier in marker.args
        ]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == "call" or report.failed or report.skipped:
        item.config._stage_one_results.setdefault(item.nodeid, []).append({
            "phase": report.when, "outcome": report.outcome, "duration_seconds": report.duration,
            "reason": str(report.longrepr) if not report.passed else None,
        })


def pytest_sessionfinish(session, exitstatus):
    name = session.config.getoption("--acceptance-report")
    if not name:
        return
    path = Path(name)
    if not path.resolve().is_relative_to((Path.cwd() / "artifacts").resolve()):
        raise ValueError("Acceptance report must be inside artifacts/")
    statuses = {}
    tests = session.config._stage_one_results
    stage = session.config.getoption("--acceptance-stage")
    if stage == 2 and not any(path.resolve().is_relative_to((Path.cwd() / folder).resolve())
                              for folder in ("artifacts/stage2", "artifacts/stage3", "artifacts/stage4")):
        raise ValueError("Stage-two acceptance cannot overwrite stage-one evidence")
    if stage == 3 and not any(path.resolve().is_relative_to((Path.cwd() / folder).resolve())
                             for folder in ("artifacts/stage3", "artifacts/stage4")):
        raise ValueError("Stage-three acceptance must be inside artifacts/stage3")
    if stage == 4 and not path.resolve().is_relative_to((Path.cwd() / "artifacts/stage4").resolve()):
        raise ValueError("Stage-four evidence must be inside artifacts/stage4")
    identifiers = ([f"D{i:02}" for i in range(1, 16)] if stage == 1 else
                   [f"M{i:02}" for i in range(1,13)] + [f"E{i:02}" for i in range(5,9)] + ["E12","A07"] if stage == 4 else
                   [f"R{i:02}" for i in range(1, 16)] if stage == 3 else
                   [f"C{i:02}" for i in range(1, 15)] + [f"E{i:02}" for i in range(1, 5)] + [f"A{i:02}" for i in range(1, 7)])
    for identifier in identifiers:
        cases = {node: outcomes for node, outcomes in tests.items()
                 if identifier in session.config._stage_one_ids.get(node, [])}
        not_executed = [node for node, ids in session.config._stage_one_ids.items()
                        if identifier in ids and node not in cases]
        outcomes = [result["outcome"] for results in cases.values() for result in results]
        status = "FAILED" if "failed" in outcomes else "NOT_EXECUTED" if not outcomes or not_executed or "skipped" in outcomes else "PASS"
        statuses[identifier] = {"status": status, "tests": cases, "not_executed_tests": not_executed}
    if stage == 4:
        deferred = {"M12":"Formal seed42/43/44 experiments not authorized in4A; only seed/CPU contracts and smoke checked",
                    "E08":"Same-key baseline and independent smoke control checks only; formal comparison matrix not run",
                    "A07":"Prediction interface and explicitly marked smoke records checked; no formal trained model available"}
        for identifier, reason in deferred.items():
            statuses[identifier]["deferred"] = reason
            if statuses[identifier]["status"] == "PASS":
                statuses[identifier]["status"] = "PARTIAL"
    from banksim.data import sha256_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spec_version": "1.0.0", "stage": stage,
                                "data_hash": sha256_file("data/raw/bs140513_032310.csv"),
                                "pytest_exit_code": int(exitstatus),
                                "acceptance": statuses, "all_tests": tests}, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def full_outputs():
    """Absent full artifacts are an explicit failure, never silently skipped."""
    base = Path("artifacts/data")
    report = json.loads((base / "quality_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    return {
        "report": report,
        "events": pd.read_parquet(base / "events_clean.parquet"),
        "panel": pd.read_parquet(base / "daily_panel.parquet"),
        "registry": pd.read_parquet(base / "customer_registry.parquet"),
        "audit": json.loads(Path("data/reference/normal_data_audit.json").read_text(encoding="utf-8")),
    }
