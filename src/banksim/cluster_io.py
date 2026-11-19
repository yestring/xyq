"""Versioned stage-two artifacts, leaving stage-one files untouched."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import SPEC_VERSION
from .allocation import RATIOS
from .clustering import ClusterBundle
from .data import DataError, sha256_file
from .data_stage import dump_json, json_default, project_relative
from .profiles import FEATURES, ProfileTransform


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=json_default).encode()).hexdigest()


def validate_config(cfg):
    if cfg["spec_version"] != SPEC_VERSION:
        raise DataError("CONFIG_MISMATCH", "Specification version differs")
    expected = {
        "forecast": {"H": 7, "L": 28, "W": 28, "C": 14, "cold_lengths": [1, 3, 7, 14]},
        "profile": {"feature_order": FEATURES, "weights": [.6, .2, .1, .1], "cv_denominator_floor": 1.0,
                    "std_ddof": 0, "winsor_quantiles": [.01, .99], "quantile_method": "linear",
                    "winsor_features": ["log_mean", "cv"], "low_info_min_days": 7, "low_info_min_positive_days": 2},
        "clustering": {"K": 3, "init": "k-means++", "n_init": 20, "max_iter": 300, "tol": .0001,
                       "random_state": 42, "fit_min_visible_days": 28, "keep_warm_all_zero": True,
                       "semantic_min_age_rank_gap": .5, "small_cluster_absolute": 20, "small_cluster_fraction": .01},
        "split": {"split_seed": 42, "warm_end": 59, "train_origin_min": 59, "train_origin_max": 112,
                  "train_label_end": 119, "validation_origin_min": 119, "validation_origin_max": 142,
                  "validation_label_end": 149, "test_origin_min": 149, "test_origin_max": 172, "test_label_end": 179,
                  "strict_id_fractions": [.7, .15, .15]},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise DataError("CONFIG_MISMATCH", "First-version protocol changed", section=section, key=key)
    for name, (conservative, aggressive) in RATIOS.items():
        if cfg["allocation"][name] != {"r_conservative": conservative, "r_aggressive": aggressive}:
            raise DataError("CONFIG_MISMATCH", "Fixed ratios differ", group=name)


def load_stage_one():
    report = json.loads(Path("artifacts/data/quality_report.json").read_text(encoding="utf-8"))
    acceptance = json.loads(Path("artifacts/acceptance_data.json").read_text(encoding="utf-8"))
    if report["status"] != "PASS" or not all(v["status"] == "PASS" for v in acceptance["acceptance"].values()):
        raise DataError("INVALID_VALUE", "Stage one is not accepted")
    source_hash = report["data_hash"]
    if sha256_file(report["source"]["path"]) != source_hash:
        raise DataError("CONFIG_MISMATCH", "Raw source hash differs")
    frames = {}
    for entry in report["artifacts"]:
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise DataError("CONFIG_MISMATCH", "Stage-one artifact hash differs", path=entry["path"])
        table = pq.read_table(entry["path"])
        if table.schema.metadata[b"data_hash"].decode() != source_hash:
            raise DataError("CONFIG_MISMATCH", "Stage-one artifact lineage differs")
        frames[Path(entry["path"]).stem] = table.to_pandas()
    return frames["events_clean"], frames["daily_panel"], frames["customer_registry"], source_hash


def metadata_for(experiment, cfg, source_hash):
    return {"spec_version": SPEC_VERSION, "stage": 2, "experiment": experiment, "data_hash": source_hash,
            "cluster_version": f"{SPEC_VERSION}-{experiment}-seed42", "rule_version": cfg["allocation"]["rule_version"],
            "fit_start": 0, "fit_cutoff": 59, "cluster_seed": 42, "split_seed": 42,
            "stability_seeds": [42, 43, 44], "config": cfg, "config_hash": fingerprint(cfg),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib", "threadpoolctl")}}


def save_json(path, value, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, {"metadata": metadata, **value})


def save_table(path, frame, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md.update(spec_version=SPEC_VERSION, data_hash=metadata["data_hash"],
              stage_metadata=json.dumps(metadata, default=json_default))
    table = table.replace_schema_metadata(md)
    pq.write_table(table, path, compression="snappy")
    if not table.equals(pq.read_table(path), check_metadata=True):
        raise DataError("INVALID_VALUE", "Stage-two Parquet round-trip differs", path=project_relative(path))
    return {"path": project_relative(path), "rows": len(frame), "sha256": sha256_file(path)}


def save_bundle(directory, bundle, profiles, seed_labels):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = bundle.metadata
    joblib.dump(bundle.model, directory / "kmeans.joblib")
    saved = {"transform": bundle.transform.to_dict(), "amount_scale": bundle.amount_scale,
             "group_map": bundle.group_map, "centers_weighted_standardized": bundle.model.cluster_centers_.tolist(),
             "summary": bundle.summary, "model_sha256": sha256_file(directory / "kmeans.joblib")}
    save_json(directory / "cluster_bundle.json", saved, metadata)
    save_json(directory / "preprocessing.json", {"transform": bundle.transform.to_dict()}, metadata)
    save_json(directory / "amount_scale.json", bundle.amount_scale, metadata)
    save_json(directory / "group_map.json", {"group_map": bundle.group_map}, metadata)
    save_table(directory / "warm_profiles.parquet", profiles, metadata)
    labels = profiles[["customer"]].copy()
    labels["raw_cluster"] = bundle.model.labels_.copy()
    labels["business_code"] = [bundle.group_map[str(int(label))]["business_code"] for label in bundle.model.labels_]
    labels["group_name"] = [bundle.group_map[str(int(label))]["group_name"] for label in bundle.model.labels_]
    labels["warm_all_zero"] = profiles.mu.eq(0)
    for seed in (43, 44):
        labels[f"raw_cluster_seed_{seed}"] = seed_labels[seed]
    save_table(directory / "warm_labels.parquet", labels, metadata)
    save_json(directory / "warm_cohort.json", {"customer_ids": profiles.customer.tolist(),
                                               "count": len(profiles), "all_zero_count": int(profiles.mu.eq(0).sum()),
                                               "label_policy": "Frozen KMeans.labels_; never replaced with AGE_ONLY assignment"}, metadata)


def load_bundle(directory, *, expected_hash=None, expected_config=None):
    directory = Path(directory)
    record = json.loads((directory / "cluster_bundle.json").read_text(encoding="utf-8"))
    metadata = record["metadata"]
    if metadata["spec_version"] != SPEC_VERSION or metadata["fit_cutoff"] != 59:
        raise DataError("CONFIG_MISMATCH", "Saved specification or cutoff differs")
    if expected_hash is not None and metadata["data_hash"] != expected_hash:
        raise DataError("CONFIG_MISMATCH", "Saved source hash differs")
    if expected_config is not None and metadata["config_hash"] != fingerprint(expected_config):
        raise DataError("CONFIG_MISMATCH", "Saved configuration differs")
    if sha256_file(directory / "kmeans.joblib") != record["model_sha256"]:
        raise DataError("CONFIG_MISMATCH", "Saved model hash differs")
    # Only load the locally generated, hash-verified model from this stage.
    model = joblib.load(directory / "kmeans.joblib")
    if not np.array_equal(model.cluster_centers_, record["centers_weighted_standardized"]):
        raise DataError("CONFIG_MISMATCH", "Saved center representations differ")
    return ClusterBundle(ProfileTransform.from_dict(record["transform"]), model, record["group_map"],
                         record["amount_scale"], record["summary"], metadata)
