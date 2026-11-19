from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import adjusted_rand_score

from banksim.cluster_io import load_bundle
from banksim.clustering import assign_profile
from banksim.data import DataError, sha256_file
from banksim.profiles import FEATURES
from banksim.splits import strict_id_split

pytestmark = pytest.mark.full
ROOT = Path("artifacts/stage2/full")


@pytest.fixture(scope="module")
def results():
    report = json.loads((ROOT / "stage2_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    return report


@pytest.mark.acceptance("C03", "C02", "E04")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_actual_warm_cohort_uses_training_ids_and_keeps_zeros(results, experiment):
    global_profiles = pd.read_parquet(ROOT / "eligibility_before_split.parquet")
    assert len(global_profiles) == 3979 and global_profiles.mu.eq(0).sum() == 177
    split = json.loads(Path(f"artifacts/splits/{experiment}.json").read_text(encoding="utf-8"))
    profiles = pd.read_parquet(ROOT / experiment / "warm_profiles.parquet")
    expected = set(global_profiles.customer) & set(split["ids"]["train"])
    assert set(profiles.customer) == expected
    assert len(profiles) == profiles.customer.nunique()
    assert profiles.mu.eq(0).sum() == global_profiles.loc[global_profiles.customer.isin(expected), "mu"].eq(0).sum()
    assert profiles.origin_t.eq(59).all() and profiles.n.eq(28).all()
    assert profiles.attribute_source_step.le(59).all()
    assert profiles.gender.isin(["F", "M"]).all() and profiles.age.ne("U").all()
    if experiment == "C":
        assert len(profiles) < 3979
        assert not set(profiles.customer) & (set(split["ids"]["validation"]) | set(split["ids"]["test"]))


@pytest.mark.acceptance("E01", "E03")
def test_saved_actual_ids_and_ordinary_exclusions(results, full_outputs):
    registry = full_outputs["registry"]
    strict = json.loads(Path("artifacts/splits/C.json").read_text(encoding="utf-8"))
    assert strict["ids"] == strict_id_split(registry.customer.tolist())
    assert [len(strict["ids"][part]) for part in ("train", "validation", "test")] == [2878, 616, 618]
    ordinary = json.loads(Path("artifacts/splits/O.json").read_text(encoding="utf-8"))
    assert ordinary["ids"]["train"] == ordinary["ids"]["validation"] == ordinary["ids"]["test"]
    # In the accepted complete panel, a<=112 is exactly ordinary training eligibility.
    assert set(ordinary["ids"]["train"]) == set(registry.loc[registry.first_seen_step.le(112), "customer"])
    assert set(ordinary["excluded_ids"]) == set(registry.customer) - set(ordinary["ids"]["train"])
    excluded = pd.read_parquet(ROOT / "O" / "excluded_customers.parquet")
    assert set(excluded.customer) == set(ordinary["excluded_ids"])


@pytest.mark.acceptance("E02", "E03", "E04")
@pytest.mark.parametrize("experiment", ["O", "C"])
@pytest.mark.parametrize("partition,bounds", [("train", (59, 112, 60, 119)), ("validation", (119, 142, 120, 149)),
                                              ("test", (149, 172, 150, 179))])
def test_all_persisted_samples_obey_id_and_time_boundaries(results, full_outputs, experiment, partition, bounds):
    path = ROOT / experiment / f"samples_{partition}.parquet"
    samples = pd.read_parquet(path)
    split = json.loads(Path(f"artifacts/splits/{experiment}.json").read_text(encoding="utf-8"))
    assert set(samples.customer) <= set(split["ids"][partition])
    assert not samples.duplicated(["experiment", "customer", "origin_t", "history_mode", "Lc"]).any()
    assert samples.origin_t.between(*bounds[:2]).all()
    assert samples.target_start.ge(bounds[2]).all() and samples.target_end.le(bounds[3]).all()
    assert samples.target_start.eq(samples.origin_t + 1).all() and samples.target_end.eq(samples.origin_t + 7).all()
    assert samples.attribute_start.equals(samples.history_start)
    assert samples.attribute_end.equals(samples.origin_t)
    first = full_outputs["registry"].set_index("customer").first_seen_step
    assert samples.history_start.ge(samples.customer.map(first)).all()
    assert samples.Tobs.eq(samples.origin_t - samples.history_start + 1).all()
    cold = samples.loc[samples.history_mode.eq("cold")]
    assert cold.Tobs.eq(cold.Lc).all()
    if partition == "train":
        assert set(samples.history_mode) == {"ordinary", "cold"}
    elif experiment == "C":
        assert set(cold.Lc) == {1, 3, 7, 14} and len(cold) == len(samples)
    else:
        assert set(samples.history_mode) == {"ordinary"}
    assert len(samples) == results["experiments"][experiment]["samples"][partition]["samples"]
    metadata = json.loads(pq.read_schema(path).metadata[b"stage_metadata"])
    assert metadata["experiment"] == experiment and metadata["fit_cutoff"] == 59
    assert metadata["data_hash"] == full_outputs["report"]["data_hash"]


@pytest.mark.acceptance("C04", "C05", "C06", "E04")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_saved_fits_match_only_the_saved_warm_training_data(results, full_outputs, experiment):
    directory = ROOT / experiment
    bundle = load_bundle(directory, expected_hash=full_outputs["report"]["data_hash"])
    profiles = pd.read_parquet(directory / "warm_profiles.parquet")
    x = profiles[FEATURES].to_numpy(dtype=float)
    lower, upper = np.quantile(x[:, [1, 2]], [.01, .99], axis=0, method="linear")
    x[:, [1, 2]] = np.clip(x[:, [1, 2]], lower, upper)
    np.testing.assert_allclose(bundle.transform.mean, x.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bundle.transform.std, x.std(axis=0, ddof=0), rtol=1e-12, atol=1e-12)
    split = json.loads(Path(f"artifacts/splits/{experiment}.json").read_text(encoding="utf-8"))
    panel = full_outputs["panel"]
    warm = panel.loc[panel.customer.isin(split["ids"]["train"]) & panel.step.le(59)
                     & panel.observed & panel.target_cents.gt(0), "target_cents"]
    assert bundle.amount_scale["a"] == np.median(warm.to_numpy(dtype=float)) / 100
    assert bundle.amount_scale["positive_daily_count"] == len(warm)
    assert bundle.model.n_clusters == 3 and bundle.model.n_init == 20 and bundle.model.max_iter == 300
    assert bundle.model.tol == .0001 and bundle.model.random_state == 42


@pytest.mark.acceptance("C03", "C08", "C09", "C11", "C14")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_model_labels_are_frozen_and_reload_consistent(results, full_outputs, experiment):
    directory = ROOT / experiment
    profiles = pd.read_parquet(directory / "warm_profiles.parquet")
    labels = pd.read_parquet(directory / "warm_labels.parquet")
    bundle = load_bundle(directory, expected_hash=full_outputs["report"]["data_hash"])
    np.testing.assert_array_equal(labels.raw_cluster, bundle.model.labels_)
    np.testing.assert_array_equal(bundle.model.predict(bundle.transform.transform(profiles)), labels.raw_cluster)
    zero_indices = profiles.index[profiles.mu.eq(0)]
    assert len(zero_indices) > 0
    for index in zero_indices[:3]:
        assert assign_profile(bundle, profiles.loc[index].to_dict())["group_status"] == "AGE_ONLY"
        assert labels.loc[index, "raw_cluster"] == bundle.model.labels_[index]
    assert all(json.loads((directory / "reload_checks.json").read_text(encoding="utf-8"))["checks"].values())
    changed_cfg = deepcopy(bundle.metadata["config"])
    changed_cfg["forecast"]["W"] = 14
    with pytest.raises(DataError) as error:
        load_bundle(directory, expected_config=changed_cfg)
    assert error.value.code == "CONFIG_MISMATCH"
    with pytest.raises(DataError):
        load_bundle(directory, expected_hash="wrong-source")


@pytest.mark.acceptance("C13", "C14", "A06")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_real_summary_and_stability_are_reported_without_performance_threshold(results, experiment):
    directory = ROOT / experiment
    diag = json.loads((directory / "cluster_diagnostics.json").read_text(encoding="utf-8"))
    summary = results["experiments"][experiment]["clusters"]
    profiles = pd.read_parquet(directory / "warm_profiles.parquet")
    labels = pd.read_parquet(directory / "warm_labels.parquet")
    assert sum(row["count"] for row in summary["clusters"]) == len(profiles)
    assert [row["business_code"] for row in summary["clusters"]] == ["A", "B", "C"]
    assert [row["age_rank_mean"] for row in summary["clusters"]] == sorted(row["age_rank_mean"] for row in summary["clusters"])
    for row in summary["clusters"]:
        members = profiles.loc[labels.raw_cluster.eq(row["raw_cluster"])]
        assert row["count"] == len(members) == sum(row["age_rank_counts"].values())
        assert row["daily_mean_amount"]["mean"] == pytest.approx(members.mu.mean(), rel=1e-12)
        assert row["small_cluster"] == (len(members) < max(20, len(profiles) * .01))
    assert summary["semantic_warning"] == any(x < .5 for x in summary["adjacent_age_rank_gaps"])
    assert -1 <= diag["silhouette_weighted_space"] <= 1
    assert diag["silhouette_sample_count"] == len(profiles)
    assert [run["seed"] for run in diag["runs"]] == [42, 43, 44]
    for pair in diag["pairwise_stability"]:
        left, right = pair["seeds"]
        left_labels = labels.raw_cluster if left == 42 else labels[f"raw_cluster_seed_{left}"]
        right_labels = labels.raw_cluster if right == 42 else labels[f"raw_cluster_seed_{right}"]
        assert pair["adjusted_rand_index"] == pytest.approx(adjusted_rand_score(left_labels, right_labels), abs=1e-12)


@pytest.mark.acceptance("A01", "A02", "A03", "A04", "A05", "A06")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_saved_real_calls_and_rules(results, experiment):
    examples = json.loads((ROOT / experiment / "allocation_examples.json").read_text(encoding="utf-8"))
    assert [(row["r_conservative"], row["r_aggressive"]) for row in examples["rule_calls"]] == [(.8, .2), (.65, .35), (.85, .15)]
    states = {row["group_status"] for row in examples["customer_calls"]}
    assert {"FULL", "AGE_ONLY", "GROUP_UNKNOWN", "NON_PERSON", "NO_HISTORY"} <= states
    for row in examples["customer_calls"]:
        if row["group_status"] in ("FULL", "AGE_ONLY"):
            assert abs(row["r_conservative"] + row["r_aggressive"] - 1) <= 1e-9
        else:
            assert row["group_name"] is row["r_conservative"] is row["r_aggressive"] is None
        assert "budget" not in row and "cash_weight" not in row


def test_stage_one_evidence_unchanged_and_stage_two_inventory_valid(results):
    baseline = json.loads(Path("artifacts/stage2/baseline.json").read_text(encoding="utf-8"))
    for path, entry in baseline.items():
        assert sha256_file(path) == entry["sha256"], path
    inventory = json.loads((ROOT / "artifact_inventory.json").read_text(encoding="utf-8"))
    for entry in inventory["files"]:
        assert sha256_file(entry["path"]) == entry["sha256"], entry["path"]
