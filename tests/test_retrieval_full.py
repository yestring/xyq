"""Audit saved real-query traces and frozen upstream evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from banksim.data import sha256_file
from banksim.retrieval_io import load_engines, read_json
from banksim.retrieval_stage import metrics


@pytest.fixture(scope="module")
def actual():
    return [json.loads(line) for line in Path("artifacts/stage3/query_results.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def engines():
    return load_engines()


@pytest.mark.acceptance("R01", "R02", "R03", "R06", "R07", "R09", "R11", "R12", "R13")
def test_every_actual_trace_is_legal_and_numerically_recomputable(actual, engines):
    for result in actual:
        e = engines[result["experiment"]]
        t, ell, customer = result["origin_t"], result["ell"], result["customer"]
        assert result["candidate_count"] == len(result["candidates"]) == len(set(result["candidate_ids"]))
        if not result["Tobs"]:
            assert result["retrieval_status"] == "NO_HISTORY" and result["candidate_count"] == 0
            continue
        np.testing.assert_array_equal(result["Q"], e.store.amounts[customer][t-ell+1:t+1])
        if result["Lc"] is not None:
            assert result["history_start"] == t-result["Lc"]+1 and result["Tobs"] == result["Lc"]
            assert result["profile"]["n"] == min(result["Lc"], 28)
            attr_source = result["attributes"]["attribute_source_step"]
            assert attr_source is None or result["history_start"] <= attr_source <= t
        assert result["retrieval_path"] == ("CROSS" if result["Tobs"] <= 14 else "SELF")
        assert result["candidate_cutoff"] == (t if result["partition"] == "train" else 119)
        by_donor = {}
        order = []
        for candidate in result["candidates"]:
            donor = candidate["donor_id"]
            start, end, vend = candidate["key_start"], candidate["key_end"], candidate["value_end"]
            assert end-start+1 == ell and vend-end == result["H"]
            assert vend <= result["candidate_cutoff"]
            x = e.store.amounts[donor]
            np.testing.assert_array_equal(candidate["K"], x[start:end+1])
            np.testing.assert_array_equal(candidate["V"], x[end+1:vend+1])
            assert np.isfinite(x[start:vend+1]).all()
            distance = float(np.mean((np.log1p(result["Q"])-np.log1p(x[start:end+1]))**2))
            assert candidate["distance"] == pytest.approx(distance, abs=1e-12)
            order.append((distance, donor, end))
            if result["retrieval_path"] == "CROSS":
                assert donor != customer and donor in e.donor_groups
                assert candidate["donor_group"] == e.donor_groups[donor]
                assert (end-(e.store.first[donor]+ell-1)) % 7 == 0
                prior = by_donor.setdefault(donor, [])
                assert len(prior) < 2 and all(start > b or a > vend for a,b in prior)
                prior.append((start, vend))
                if result["selection_status"] == "SAME_GROUP":
                    assert result["group_status"] == "FULL" and candidate["donor_group"] == result["raw_cluster"]
            else:
                assert donor == customer and start >= result["history_start"] and vend <= t-ell
                assert (end-(result["history_start"]+ell-1)) % 7 == 0
        assert order == sorted(order)
        if result["candidate_count"]:
            weights = np.array([c["weight"] for c in result["candidates"]])
            distances = np.array([c["distance"] for c in result["candidates"]])
            expected = np.exp(-(distances-distances.min()))
            expected /= expected.sum()
            np.testing.assert_allclose(weights, expected, atol=1e-14)
            np.testing.assert_allclose(result["R"], weights @ np.array([c["V"] for c in result["candidates"]]), rtol=1e-12)
            np.testing.assert_allclose(result["r_log"], np.log1p(result["R"]), rtol=1e-12)
            assert result["reference_available"] == result["r_mask"] == 1
        else:
            assert result["R"] == result["r_log"] == [0]*7
            assert result["reference_available"] == result["r_mask"] == 0


@pytest.mark.acceptance("R06", "R14")
@pytest.mark.parametrize("experiment", ["O", "C"])
def test_actual_frozen_donor_cohort_and_fresh_reload(actual, engines, experiment):
    e = engines[experiment]
    labels = pd.read_parquet(f"artifacts/stage2/full/{experiment}/warm_labels.parquet")
    cohort = read_json(f"artifacts/stage2/full/{experiment}/warm_cohort.json")
    split = read_json(f"artifacts/splits/{experiment}.json")["ids"]
    assert set(e.donor_groups) == set(cohort["customer_ids"]) <= set(split["train"])
    assert e.donor_groups == dict(zip(labels.customer, labels.raw_cluster))
    if experiment == "C":
        assert not set(e.donor_groups) & (set(split["validation"]) | set(split["test"]))
    for saved in (r for r in actual if r["experiment"] == experiment and r["cohort"] == "main_sample"):
        current = e.retrieve(saved["customer"], saved["origin_t"], Lc=saved["Lc"],
                             partition=saved["partition"], method=saved["method"])
        assert current["cache_hit"] and current["candidate_ids"] == saved["candidate_ids"]
        assert current["R"] == saved["R"] and current["versions"] == saved["versions"]


@pytest.mark.acceptance("R10", "R14", "R15")
def test_actual_sample_metrics_resource_bounds_and_separate_expansion(actual):
    report = read_json("artifacts/stage3/sample_report.json")
    assert report["kind"] == "small_reproducible_query_batch_not_full_experiment"
    assert report["metrics"] == metrics(actual)
    assert report["total_calls"] == len(actual)
    assert all(r["replay_cache_hit"] and r["replay_equal"] for r in actual)
    assert report["max_distance_block_rows"] <= 2048
    assert report["max_candidates_scored_for_one_query"] > report["max_distance_block_rows"]
    assert {r["selection_status"] for r in actual} >= {"SAME_GROUP", "ALL_DONORS", "SELF", "EXPANDED_ALL"}
    for r in actual:
        if r["cohort"] == "expansion_sensitivity_m1500":
            assert r["parameters"]["m"] == 1500 and r["expanded_all"]
            assert r["same_group_accepted_count"] < 1500
        else:
            assert r["parameters"]["m"] == 5


def test_all_upstream_evidence_unchanged():
    baseline = read_json("artifacts/stage3/baseline.json")
    for path, saved in baseline.items():
        assert sha256_file(path) == saved["sha256"], path
    assert Path("DEVELOPMENT_PROGRESS.md").read_bytes().startswith(Path("artifacts/stage3/progress-before.txt").read_bytes())
