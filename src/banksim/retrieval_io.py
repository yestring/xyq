"""Load accepted, hash-checked upstream artifacts without refitting anything."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .allocation import NAMES
from .candidates import HistoryStore
from .cluster_io import fingerprint, load_bundle, load_stage_one
from .data import DataError, sha256_file, validate_records
from .panel import build_daily_panel
from .profiles import ProfileSource, ProfileTransform, apply_feature_weights
from .retrieval import RetrievalEngine, RetrievalParams
from .retrieval_cache import RetrievalCache


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def implementation_version():
    from .retrieval_version import semantic_version
    return semantic_version()


def load_engines(cache_root="artifacts/stage3/cache"):
    cfg = read_json("config/defaults.json")
    acceptance = read_json("artifacts/stage2/acceptance-cluster.json")
    if not all(x["status"] == "PASS" for x in acceptance["acceptance"].values()):
        raise DataError("CONFIG_MISMATCH", "Stage two has not passed acceptance")
    for entry in read_json("artifacts/stage2/full/artifact_inventory.json")["files"]:
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise DataError("CONFIG_MISMATCH", "Frozen stage-two artifact changed", path=entry["path"])
    raw, panel, registry, data_hash = load_stage_one()
    source, store = ProfileSource(raw, panel), HistoryStore(panel)
    cache = RetrievalCache(cache_root) if cache_root else None
    params = RetrievalParams(**{key: cfg["forecast"][key] for key in ("H", "L", "C", "W")},
                             m=cfg["retrieval"]["top_m"], tau=cfg["retrieval"]["tau"],
                             stride=cfg["retrieval"]["candidate_stride"],
                             max_per_donor=cfg["retrieval"]["max_windows_per_cross_donor"],
                             eval_cutoff=cfg["retrieval"]["main_eval_cutoff"])
    engines = {}
    for experiment in ("O", "C"):
        base = Path("artifacts/stage2/full") / experiment
        bundle = load_bundle(base, expected_hash=data_hash, expected_config=cfg)
        split = read_json(f"artifacts/splits/{experiment}.json")
        cohort = read_json(base / "warm_cohort.json")
        labels = pd.read_parquet(base / "warm_labels.parquet")
        if (bundle.metadata["experiment"] != experiment or split["experiment"] != experiment
                or split["metadata"]["data_hash"] != data_hash
                or cohort["customer_ids"] != labels.customer.tolist()
                or labels.customer.duplicated().any()
                or not np.array_equal(labels.raw_cluster, bundle.model.labels_)):
            raise DataError("CONFIG_MISMATCH", "Frozen donor labels or experiment lineage differ")
        if experiment == "C" and set(labels.customer) & (set(split["ids"]["validation"]) | set(split["ids"]["test"])):
            raise DataError("CONFIG_MISMATCH", "Held-out IDs cannot be cross-customer donors")
        versions = {"spec_version": cfg["spec_version"], "data_hash": data_hash,
                    "panel_hash": sha256_file("artifacts/data/daily_panel.parquet"),
                    "config_version": fingerprint(cfg), "cluster_version": bundle.metadata["cluster_version"],
                    "cluster_bundle_hash": sha256_file(base / "cluster_bundle.json"),
                    "model_hash": sha256_file(base / "kmeans.joblib"),
                    "labels_file_hash": sha256_file(base / "warm_labels.parquet"),
                    "split_hash": sha256_file(f"artifacts/splits/{experiment}.json"),
                    "implementation_version": implementation_version(), "fit_cutoff": 59}
        engines[experiment] = RetrievalEngine(experiment, source, bundle,
                                             dict(zip(labels.customer, labels.raw_cluster.astype(int))),
                                             split["ids"]["train"], versions, params=params, cache=cache, store=store)
    return engines


def toy_engine(cache=None):
    example = read_json("examples/toy_walkthrough.json")
    views = validate_records(example["raw_events"], coverage=tuple(example["coverage"]))
    panel, _ = build_daily_panel(views, coverage=tuple(example["coverage"]))
    weights = np.array([.6, .2, .1, .1])
    transform = ProfileTransform(np.zeros(4), np.ones(4), np.array([-1e6, -1e6]),
                                 np.array([1e6, 1e6]), np.zeros(4, dtype=bool), weights)
    centers = apply_feature_weights([[age, 0, 0, 0] for age in example["query"]["assumed_cluster_age_centers"]], weights)
    bundle = SimpleNamespace(transform=transform, model=SimpleNamespace(cluster_centers_=centers),
                             group_map={str(i): {"business_code": "ABC"[i], "group_name": NAMES[i]} for i in range(3)},
                             summary={"semantic_warning": False})
    versions = {"spec_version": "1.0.0", "data_hash": sha256_file("examples/toy_walkthrough.json"),
                "cluster_version": "explicit_toy_assumptions_not_fitted", "implementation_version": implementation_version()}
    return RetrievalEngine("TOY", ProfileSource(views.raw_events, panel), bundle, {"D1": 0, "D2": 1},
                           ["D1", "D2"], versions, params=RetrievalParams(H=2, L=2, stride=1), cache=cache)
