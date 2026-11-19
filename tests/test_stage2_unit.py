from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from banksim.allocation import NAMES, map_allocation
from banksim.cluster_io import validate_config
from banksim.clustering import (assign_profile, cluster_summary, describe_customer, fit_bundle,
                                fit_kmeans, name_clusters)
from banksim.data import DataError, validate_records
from banksim.panel import build_daily_panel
from banksim.profiles import (FEATURES, ProfileSource, ProfileTransform, apply_feature_weights,
                              fit_amount_scale, profile_values)
from banksim.splits import SampleIndex, build_id_split, origin_range, strict_id_split


@pytest.fixture
def cfg():
    return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))


def event(customer="T", step=0, amount="10", age="2", gender="F", fraud=0):
    return dict(step=step, customer=customer, age=age, gender=gender, zipcodeOri="28007", merchant="M",
                zipMerchant="28007", category="es_food", amount=amount, fraud=fraud)


def fixture_source(records):
    views = validate_records(records)
    panel, registry = build_daily_panel(views)
    return ProfileSource(views.raw_events, panel), panel, registry, views.raw_events


def known_bundle():
    weights = np.array([.6, .2, .1, .1])
    transform = ProfileTransform(np.zeros(4), np.ones(4), np.array([-1e6, -1e6]),
                                 np.array([1e6, 1e6]), np.zeros(4, dtype=bool), weights)
    centers = apply_feature_weights([[6, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]], weights)
    mapping = {"0": {"business_code": "C", "group_name": NAMES[2]},
               "1": {"business_code": "A", "group_name": NAMES[0]},
               "2": {"business_code": "B", "group_name": NAMES[1]}}
    return SimpleNamespace(transform=transform, model=SimpleNamespace(cluster_centers_=centers),
                           group_map=mapping, summary={"semantic_warning": False})


@pytest.mark.acceptance("C01")
def test_profile_exact_formula():
    profile = profile_values([10, 0, 30, 20], "2", "F")
    np.testing.assert_allclose([profile[k] for k in ["mu", "sigma", "cv", "positive_day_ratio", "log_mean"]],
                               [15, np.sqrt(125), np.sqrt(125) / 15, .75, np.log(16)], rtol=1e-12, atol=1e-12)
    low = profile_values([0, .2], "2", "M")
    assert low["cv"] == low["sigma"]  # denominator floor is one original amount unit.


@pytest.mark.acceptance("C02")
def test_zero_days_and_zero_amount_transactions():
    source, _, _, _ = fixture_source([event(step=0, amount="0"), event(step=3, amount="0")])
    profile = source.describe("T", 5)
    assert [profile[k] for k in ("mu", "sigma", "cv", "positive_day_ratio", "positive_days")] == [0] * 5
    assert profile["Tobs"] == profile["n"] == 6


@pytest.mark.acceptance("C05")
def test_winsor_linear_quantiles_and_constant_features():
    profiles = pd.DataFrame({"age_rank": [2] * 100, "log_mean": np.arange(100),
                             "cv": [0.] * 100, "positive_day_ratio": [.5] * 100})
    fitted = ProfileTransform.fit(profiles)
    np.testing.assert_allclose(fitted.lower, [.99, 0], rtol=1e-12)
    np.testing.assert_allclose(fitted.upper, [98.01, 0], rtol=1e-12)
    assert fitted.constant.tolist() == [True, False, True, True]
    changed = pd.DataFrame([[6, 1e9, 1e9, 1]], columns=FEATURES)
    transformed = fitted.transform(changed)[0]
    assert transformed[[0, 2, 3]].tolist() == [0, 0, 0]
    assert transformed[1] == pytest.approx((98.01 - fitted.mean[1]) / fitted.std[1] * np.sqrt(.2), rel=1e-12)
    diverse = pd.DataFrame([[0, 0, 0, 0], [3, 1, 1, .5], [6, 2, 2, 1]], columns=FEATURES)
    t = ProfileTransform.fit(diverse)
    changed = pd.DataFrame([[12, 2, 2, 2]], columns=FEATURES)
    assert t.transform(changed)[0, 0] == pytest.approx((12-t.mean[0])/t.std[0]*np.sqrt(.6))
    assert t.transform(changed)[0, 3] == pytest.approx((2-t.mean[3])/t.std[3]*np.sqrt(.1))


@pytest.mark.acceptance("C06")
def test_sqrt_feature_weight_squared_distance():
    weighted = apply_feature_weights([1, 2, 3, 4], [.6, .2, .1, .1])
    assert float(weighted @ weighted) == pytest.approx(3.9, abs=1e-12)
    assert not np.isclose(weighted @ weighted, .77)


@pytest.mark.acceptance("C07")
@pytest.mark.parametrize("age,expected", [("0", 0), ("6", 6), ("U", None)])
def test_age_is_ordinal_and_unknown_is_not_seven(age, expected):
    profile = profile_values([1] * 7, age, "F")
    assert profile["age_rank"] == expected
    assert not ({"income", "assets", "marriage", "retired", "exact_age"} & profile.keys())


@pytest.mark.acceptance("C08", "A01", "A04")
@pytest.mark.parametrize("age,raw_label,name,ratios", [("0", 1, NAMES[0], (.8, .2)),
                                                    ("2", 2, NAMES[1], (.65, .35)),
                                                    ("6", 0, NAMES[2], (.85, .15))])
def test_full_assignment_and_three_actual_ratio_calls(age, raw_label, name, ratios):
    output = assign_profile(known_bundle(), profile_values([1] * 7, age, "F"))
    assert output["group_status"] == "FULL"
    assert output["raw_cluster"] == raw_label and output["group_name"] == name
    assert (output["r_conservative"], output["r_aggressive"]) == ratios
    assert abs(sum(ratios) - 1) <= 1e-9 and all(0 <= x <= 1 for x in ratios)
    assert output["rule_version"] == "1.0.0"


@pytest.mark.acceptance("C09", "A02")
@pytest.mark.parametrize("values", [[1e8], [0, 0, 0, 0, 0, 0, 100]])
def test_age_only_ignores_behavior_and_breaks_ties_by_business_code(values):
    result = assign_profile(known_bundle(), profile_values(values, "1", "F"))
    assert result["group_status"] == result["allocation_status"] == "AGE_ONLY"
    assert result["business_code"] == "A" and result["raw_cluster"] == 1
    assert not result["same_group_filter_allowed"]
    assert result["note"] == "仅按年龄暂定"
    assert (result["r_conservative"], result["r_aggressive"]) == (.8, .2)


@pytest.mark.acceptance("C10", "A03")
@pytest.mark.parametrize("age,gender,status", [("U", "F", "GROUP_UNKNOWN"), ("2", "U", "GROUP_UNKNOWN"),
                                               ("2", "E", "NON_PERSON"), ("U", "E", "NON_PERSON")])
def test_unknown_and_enterprise_no_fourth_cluster(age, gender, status):
    result = assign_profile(known_bundle(), profile_values([1] * 8, age, gender))
    assert result["group_status"] == status
    assert result["group_name"] is result["raw_cluster"] is result["r_conservative"] is result["r_aggressive"] is None


@pytest.mark.acceptance("C10", "A03", "E04")
def test_cold_attributes_cannot_be_restored_from_hidden_or_future_raw():
    source, _, _, _ = fixture_source([event(step=0), event(step=5, fraud=1), event(step=7, age="6")])
    unknown = describe_customer(known_bundle(), source, "T", 4, Lc=1)
    assert unknown["group_status"] == "GROUP_UNKNOWN"
    assert unknown["profile"]["age"] == unknown["profile"]["gender"] == "U"
    visible_fraud = describe_customer(known_bundle(), source, "T", 5, Lc=1)
    assert visible_fraud["group_status"] == "AGE_ONLY"
    assert visible_fraud["profile"]["mu"] == 0 and visible_fraud["attribute_source_step"] == 5
    assert describe_customer(known_bundle(), source, "T", -1)["group_status"] == "NO_HISTORY"


@pytest.mark.acceptance("C11")
def test_business_order_and_both_tie_breakers():
    rows = pd.DataFrame({"age_rank": [6, 2, 3], "mu": [1, 2, 3]})
    mapping = name_clusters(rows, [0, 1, 2])
    assert [mapping[str(label)]["business_code"] for label in [1, 2, 0]] == ["A", "B", "C"]
    rows = pd.DataFrame({"age_rank": [2, 2, 2], "mu": [50, 10, 10]})
    mapping = name_clusters(rows, [0, 1, 2])
    assert [mapping[str(label)]["business_code"] for label in [1, 2, 0]] == ["A", "B", "C"]


@pytest.mark.acceptance("C12", "A03")
def test_too_few_distinct_profiles_blocks_experiment(cfg):
    profiles = pd.DataFrame([profile_values([1]*28, "2", "F")] * 3)
    with pytest.raises(DataError) as caught:
        fit_kmeans(profiles, ProfileTransform.fit(profiles), cfg)
    assert caught.value.code == "CLUSTER_FIT_FAILED"
    source, _, _, _ = fixture_source([event()])
    failed = describe_customer(None, source, "T", 59)
    assert failed["group_status"] == "CLUSTER_FIT_FAILED"
    assert failed["prediction_status"] == "BLOCKED"
    assert failed["predicted_daily"] is failed["group_name"] is failed["r_conservative"] is None


@pytest.mark.acceptance("C12")
def test_two_nonempty_clusters_also_fail(monkeypatch, cfg):
    profiles = pd.DataFrame([profile_values([i + 1] * 28, str(i), "F") for i in range(4)])
    class Collapsed:
        def __init__(self, **kwargs):
            pass
        def fit(self, values):
            self.labels_ = np.array([0, 0, 1, 1])
    monkeypatch.setattr("banksim.clustering.KMeans", Collapsed)
    with pytest.raises(DataError) as caught:
        fit_kmeans(profiles, ProfileTransform.fit(profiles), cfg)
    assert caught.value.code == "CLUSTER_FIT_FAILED"


@pytest.mark.acceptance("C13", "A06")
def test_warning_thresholds_report_without_moving_members(cfg):
    profiles = pd.DataFrame([profile_values([1]*28, "2", "F"), profile_values([2]*28, "2", "F"),
                             profile_values([3]*28, "2", "F")])
    labels = np.array([0, 1, 2])
    result = cluster_summary(profiles, labels, np.zeros((3, 4)), name_clusters(profiles, labels), cfg)
    assert result["semantic_warning"] and result["small_cluster"]
    assert result["small_cluster_threshold"] == 20
    assert result["adjacent_age_rank_gaps"] == [0, 0]
    assert [row["count"] for row in result["clusters"]] == [1, 1, 1]
    assert "完整阶段画像尚未验证" in result["interpretation"]
    assert "稳健投资比例不代表现金储备" in result["interpretation"]


@pytest.mark.acceptance("E01")
def test_exact_strict_id_protocol_and_fixed_seed():
    ids = [f"C{i}" for i in range(4112)]
    result = strict_id_split(reversed(ids))
    assert [len(result[key]) for key in ["train", "validation", "test"]] == [2878, 616, 618]
    assert not set(result["train"]) & set(result["test"])
    assert not set(result["train"]) & set(result["validation"])
    assert not set(result["test"]) & set(result["validation"])
    assert set(sum(result.values(), [])) == set(ids)
    expected = np.array(sorted(ids), dtype=object)
    np.random.default_rng(42).shuffle(expected)
    assert sum(result.values(), []) == expected.tolist()


@pytest.mark.acceptance("E02", "E03")
def test_origin_boundaries_and_ordinary_late_customer_admission(cfg):
    _, panel, registry, _ = fixture_source([event("EARLY", 0), event("MID", 70), event("LATE", 113), event("NEW", 135)])
    index = SampleIndex(panel, registry)
    split = build_id_split(index, "O", cfg)
    assert split["ids"]["train"] == ["EARLY", "MID"]
    assert split["ids"]["validation"] == split["ids"]["test"] == ["EARLY", "MID"]
    assert split["excluded_ids"] == ["LATE", "NEW"]
    expected = {"train": (59, 112, 60, 119), "validation": (119, 142, 120, 149), "test": (149, 172, 150, 179)}
    for part, bounds in expected.items():
        samples = index.samples("O", part, split["ids"][part], cfg)
        assert (samples.origin_t.min(), samples.origin_t.max(), samples.target_start.min(), samples.target_end.max()) == bounds
        assert samples.target_start.eq(samples.origin_t + 1).all()
        assert samples.target_end.eq(samples.origin_t + 7).all()
        assert samples.loc[samples.customer.eq("MID"), "origin_t"].min() >= 70
        assert samples.attribute_start.equals(samples.history_start)
        assert samples.attribute_end.equals(samples.origin_t)
    # C validation/test have exactly the four short-history modes.
    cold = index.samples("C", "validation", ["EARLY"], cfg)
    assert set(cold.Lc) == {1, 3, 7, 14} and set(cold.history_mode) == {"cold"}
    assert cold.Tobs.eq(cold.Lc).all()


@pytest.mark.acceptance("E02")
def test_samples_reject_observation_gaps(cfg):
    _, panel, registry, _ = fixture_source([event()])
    panel.loc[panel.step.eq(119), "observed"] = False
    index = SampleIndex(panel, registry)
    train = index.samples("O", "train", ["T"], cfg)
    assert 112 not in train.origin_t.to_list()  # last training label would include the gap.
    assert index.samples("O", "validation", ["T"], cfg).empty
    assert not index.legal("T", 173, 180)


@pytest.mark.acceptance("C04", "E04")
def test_future_and_holdout_mutation_cannot_change_any_fit(cfg):
    records = [event(f"C{i}", day, str((i+1)*2 + day % (i+2)), age=str(i % 7))
               for i in range(12) for day in range(0, 80)]
    source, panel, _, raw = fixture_source(records)
    train_ids = [f"C{i}" for i in range(9)]
    base, profiles = fit_bundle(source, panel, train_ids, cfg, {})
    altered_raw, altered_panel = raw.copy(), panel.copy()
    hidden = altered_raw.step.ge(60) | ~altered_raw.customer.isin(train_ids)
    altered_raw.loc[hidden, "age"] = "U"
    altered_raw.loc[hidden, "gender"] = "E"
    hidden_panel = altered_panel.step.ge(60) | ~altered_panel.customer.isin(train_ids)
    altered_panel.loc[hidden_panel, "target_cents"] = 99999999
    changed, changed_profiles = fit_bundle(ProfileSource(altered_raw, altered_panel), altered_panel, train_ids, cfg, {})
    pd.testing.assert_frame_equal(profiles, changed_profiles)
    assert base.transform.to_dict() == changed.transform.to_dict()
    assert base.amount_scale == changed.amount_scale
    np.testing.assert_array_equal(base.model.cluster_centers_, changed.model.cluster_centers_)
    np.testing.assert_array_equal(base.model.labels_, changed.model.labels_)
    ordinary, _ = fit_bundle(source, panel, [f"C{i}" for i in range(12)], cfg, {})
    assert ordinary.transform is not base.transform
    assert ordinary.transform.to_dict() != base.transform.to_dict()


@pytest.mark.acceptance("C04")
def test_amount_scale_uses_positive_daily_totals_all_training_ids_and_fallback():
    _, panel, _, _ = fixture_source([event("A", 0, "10"), event("A", 0, "30"), event("A", 1, "20"),
                                    event("B", 0, "100", gender="E"), event("A", 60, "999")])
    scale = fit_amount_scale(panel, ["A", "B"])
    assert scale["a"] == 40 and scale["positive_daily_count"] == 3
    assert fit_amount_scale(panel, ["A"])["a"] == 30
    assert fit_amount_scale(panel, ["MISSING"])["a"] == 1


@pytest.mark.acceptance("A05", "A06")
def test_fixed_mapping_ignores_forecast_and_emits_only_two_ratios():
    group = {"group_status": "FULL", "group_name": NAMES[1], "predicted_total": 1}
    result = map_allocation(group)
    group["predicted_total"] = 1e12
    assert map_allocation(group) == result
    assert set(result) == {"group_name", "allocation_status", "r_conservative", "r_aggressive", "rule_version"}


@pytest.mark.acceptance("A04", "A05")
def test_configuration_cannot_silently_change_fixed_business_rules(cfg):
    validate_config(cfg)
    changed = deepcopy(cfg)
    changed["allocation"][NAMES[0]]["r_conservative"] = .7
    with pytest.raises(DataError) as caught:
        validate_config(changed)
    assert caught.value.code == "CONFIG_MISMATCH"
