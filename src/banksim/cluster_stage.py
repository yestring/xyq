"""Stage two only: python src/banksim/cluster_stage.py (run at project root)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from banksim.allocation import NAMES, map_allocation
from banksim.cluster_io import (fingerprint, load_bundle, load_stage_one, metadata_for, save_bundle,
                               save_json, save_table, validate_config)
from banksim.clustering import describe_customer, fit_bundle, stability_diagnostics
from banksim.data import DataError, sha256_file
from banksim.data_stage import dump_json, json_default, project_relative
from banksim.profiles import ProfileSource, warm_profiles
from banksim.splits import SampleIndex, build_id_split, sample_summary


def call_examples(bundle, source, profiles, raw, registry):
    requests = []
    for label in range(3):
        choices = profiles.loc[(bundle.model.labels_ == label) & profiles.positive_days.ge(2)]
        if len(choices):
            requests.append((str(choices.iloc[0].customer), 59, None))
    zeros = profiles.loc[profiles.positive_days.lt(2)]
    if len(zeros):
        requests.append((str(zeros.iloc[0].customer), 59, None))
    visible = raw.loc[raw.step.le(59)]
    for mask in (visible.age.eq("U") | visible.gender.eq("U"), visible.gender.eq("E")):
        if mask.any():
            requests.append((str(visible.loc[mask].iloc[0].customer), 59, None))
    late = registry.loc[registry.first_seen_step.gt(0)].iloc[0]
    requests.append((str(late.customer), int(late.first_seen_step) - 1, None))
    for customer in profiles.customer.head(100):
        example = describe_customer(bundle, source, customer, 119, Lc=1)
        if example["group_status"] == "GROUP_UNKNOWN":
            requests.append((str(customer), 119, 1))
            break
    return [{"example_kind": "real_data_asof_call", **describe_customer(bundle, source, customer, t, Lc=lc)}
            for customer, t, lc in requests]


def execute(cfg, output):
    validate_config(cfg)
    raw, panel, registry, data_hash = load_stage_one()
    source, index = ProfileSource(raw, panel), SampleIndex(panel, registry)
    all_eligible = warm_profiles(source, registry.customer, cfg)
    top_metadata = metadata_for("O_and_C_independent", cfg, data_hash)
    report = {"status": "RUNNING", "mode": "full", "eligibility_before_split": {
        "count": len(all_eligible), "all_zero_count": int(all_eligible.mu.eq(0).sum()),
        "purpose": "Descriptive eligibility audit only; never used to fit C transforms or centers"}, "experiments": {}}
    save_table(output / "eligibility_before_split.parquet", all_eligible, top_metadata)
    for experiment in ("O", "C"):
        print(f"{experiment}: building ID split and legal windows", flush=True)
        directory = output / experiment
        directory.mkdir(parents=True, exist_ok=True)
        metadata = metadata_for(experiment, cfg, data_hash)
        split = build_id_split(index, experiment, cfg)
        metadata["training_ids_hash"] = fingerprint(split["ids"]["train"])
        summary = {"id_counts": {part: len(ids) for part, ids in split["ids"].items()},
                   "excluded_customer_count": len(split["excluded_ids"]), "samples": {}}
        for partition, ids in split["ids"].items():
            samples = index.samples(experiment, partition, ids, cfg)
            summary["samples"][partition] = sample_summary(samples, ids)
            save_table(directory / f"samples_{partition}.parquet", samples, metadata)
            print(f"{experiment} {partition}: {len(samples)} windows, {samples.customer.nunique()} customers", flush=True)
            del samples
        split["sample_summary"] = summary["samples"]
        split["sample_directory"] = project_relative(directory)
        save_json(Path("artifacts/splits") / f"{experiment}.json", split, metadata)
        excluded = registry.loc[registry.customer.isin(split["excluded_ids"]), ["customer", "first_seen_step"]].copy()
        excluded["seen_by_train_label_end"] = excluded.first_seen_step.le(119)
        excluded["reason"] = "No legal ordinary training window"
        save_table(directory / "excluded_customers.parquet", excluded, metadata)
        print(f"{experiment}: fitting warmup transforms and K-means (seeds 42/43/44)", flush=True)
        bundle, profiles = fit_bundle(source, panel, split["ids"]["train"], cfg, metadata)
        diagnostics, seed_labels = stability_diagnostics(bundle, profiles, cfg)
        save_bundle(directory, bundle, profiles, seed_labels)
        save_json(directory / "cluster_diagnostics.json", diagnostics, metadata)
        examples = call_examples(bundle, source, profiles, raw, registry)
        rule_calls = [map_allocation({"group_name": name, "group_status": "FULL"}) for name in NAMES]
        save_json(directory / "allocation_examples.json", {"customer_calls": examples,
                  "rule_calls_kind": "Actual fixed-rule calls, not customer classifications", "rule_calls": rule_calls}, metadata)
        loaded = load_bundle(directory, expected_hash=data_hash, expected_config=cfg)
        roundtrip = {
            "transforms_equal": bool(np.array_equal(bundle.transform.transform(profiles), loaded.transform.transform(profiles))),
            "centers_equal": bool(np.array_equal(bundle.model.cluster_centers_, loaded.model.cluster_centers_)),
            "raw_labels_equal": bool(np.array_equal(bundle.model.labels_, loaded.model.labels_)),
            "predict_labels_equal": bool(np.array_equal(bundle.model.predict(bundle.transform.transform(profiles)),
                                                         loaded.model.predict(loaded.transform.transform(profiles)))),
            "amount_scale_equal": bundle.amount_scale == loaded.amount_scale,
            "group_map_equal": bundle.group_map == loaded.group_map,
            "calls_equal": examples == call_examples(loaded, source, profiles, raw, registry),
        }
        if not all(roundtrip.values()):
            raise DataError("CONFIG_MISMATCH", "Reloaded cluster artifacts changed results", checks=roundtrip)
        save_json(directory / "reload_checks.json", {"checks": roundtrip}, metadata)
        summary.update(warm_fit_count=len(profiles), warm_all_zero_count=int(profiles.mu.eq(0).sum()),
                       amount_scale=bundle.amount_scale, clusters=bundle.summary,
                       silhouette_weighted_space=diagnostics["silhouette_weighted_space"],
                       stability=diagnostics["pairwise_stability"], reload_checks=roundtrip,
                       split_path=f"artifacts/splits/{experiment}.json", artifact_directory=project_relative(directory))
        report["experiments"][experiment] = summary
        print(f"{experiment}: fit {len(profiles)} customers (zero={int(profiles.mu.eq(0).sum())}), a={bundle.amount_scale['a']}", flush=True)
    report["status"] = "PASS"
    save_json(output / "stage2_report.json", report, top_metadata)
    inventory = [{"path": project_relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"]
    save_json(output / "artifact_inventory.json", {"files": inventory}, top_metadata)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--output", default="artifacts/stage2/full")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.resolve().is_relative_to((Path.cwd() / "artifacts/stage2").resolve()):
        print("INVALID_VALUE: stage-two output must be inside artifacts/stage2/")
        return 2
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        report = execute(json.loads(Path(args.config).read_text(encoding="utf-8")), output)
    except DataError as exc:
        dump_json(output / "stage2_failure.json", {"status": "FAILED", "stage": 2, "error": exc.as_dict(),
                                                   "group_name": None, "predicted_daily": None,
                                                   "r_conservative": None, "r_aggressive": None})
        print(json.dumps(exc.as_dict(), default=json_default))
        return 2
    print(json.dumps({"status": report["status"], "elapsed_seconds": round(time.perf_counter() - started, 3),
                      "report": project_relative(output / "stage2_report.json")}, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
