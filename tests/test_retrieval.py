"""Executable stage-three boundaries and independent exact-search oracle."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from banksim.candidates import HistoryStore, build_bank, reference_weights
from banksim.cluster_io import fingerprint
from banksim.data import DataError, validate_records
from banksim.panel import build_daily_panel
from banksim.profiles import ProfileSource
from banksim.retrieval import RetrievalEngine, RetrievalParams
from banksim.retrieval_cache import RetrievalCache
from banksim.retrieval_io import read_json, toy_engine


def event(customer, step, amount=10, age="2", gender="F"):
    return dict(customer=customer, step=step, amount=str(amount), age=age, gender=gender,
                zipcodeOri="28007", merchant="M", zipMerchant="28007", category="es_food", fraud=0)


def engine(records=None, *, donors=None, params=None, cache=None, experiment="TEST", versions=None):
    if records is None:
        records = [event(c, t) for c in ("T", "D1", "D2", "D3") for t in range(180)]
    views = validate_records(records)
    panel, _ = build_daily_panel(views)
    groups = {"D1": 0, "D2": 1, "D3": 2} if donors is None else donors
    return RetrievalEngine(experiment, ProfileSource(views.raw_events, panel), toy_engine().bundle,
                           groups, list(groups), versions or {"data_hash": fingerprint(records), "cluster_version": "fixture"},
                           params=params, cache=cache)


@pytest.mark.acceptance("R01")
@pytest.mark.parametrize("t,expected", [(-1, "NO_HISTORY"), (0, "CROSS"), (13, "CROSS"), (14, "SELF")])
def test_route_and_zero_days(t, expected):
    e = engine([event("T", 1, 0), event("D1", 0)], donors={"D1": 0})
    r = e.retrieve("T", t+1)
    assert (r["retrieval_status"] if expected == "NO_HISTORY" else r["retrieval_path"]) == expected
    assert r["Tobs"] == t+1
    assert r["Q"] == [0.] * max(0, t+1)
    assert not any("predicted" in key for key in r)


@pytest.mark.acceptance("R02", "R03", "R05")
def test_grid_cutoff_and_continuity():
    e = engine([event("D", 3)], donors={"D": 0})
    make = lambda **kw: build_bank(e.store, ["D"], {"D": 0}, identity={}, **kw)
    b = make(ell=28, H=7, stride=7, cutoff=62)
    assert b.key_end.tolist() == [30, 37, 44, 51]
    assert make(ell=2, H=2, stride=1, cutoff=10).key_end[-1] == 8
    assert make(ell=2, H=2, stride=1, cutoff=10, value_limit=9).key_end[-1] == 7
    for ell in range(1, 29):
        b = make(ell=ell, H=7, stride=7, cutoff=119)
        assert b.key_end[0] == 3+ell-1
        assert all(int(s)+7 <= 119 for s in b.key_end)
    # Missing donor dates are not silently interpolated into complete K/V.
    damaged = e.source.panels["D"].loc[lambda x: x.step.ne(5)]
    store = HistoryStore(damaged)
    b = build_bank(store, ["D"], {"D": 0}, ell=2, H=2, stride=1, cutoff=10, identity={})
    assert b.key_end.tolist() == [7, 8]


@pytest.mark.acceptance("R02", "R03", "R04", "R05", "R11")
def test_self_counts_and_frozen_cutoff():
    e = engine()
    for t, ends in [(61, []), (62, [27]), (89, [27, 34, 41, 48])]:
        r = e.retrieve("T", t)
        assert [c["key_end"] for c in r["candidates"]] == ends
        assert r["retrieval_path"] == "SELF"
        assert r["scored_candidate_count"] == len(ends)
        assert all(c["value_end"] <= t-r["ell"] for c in r["candidates"])
    e1 = engine(params=RetrievalParams(stride=1))
    r = e1.retrieve("T", 89)
    assert r["scored_candidate_count"] == 28  # s=27..54, V end=34..61 inclusive.
    assert max(b.key_end.max() for b in e1._banks.values())+7 == 89-28
    for partition, t in [("train", 112), ("validation", 142), ("test", 172)]:
        r = e.retrieve("T", t, Lc=3, partition=partition)
        assert r["candidate_cutoff"] == (t if partition == "train" else 119)
        assert all(c["value_end"] <= r["candidate_cutoff"] for c in r["candidates"])
    b = build_bank(e.store, ["D1"], {"D1": 0}, ell=28, H=7, stride=7, cutoff=59, identity={})
    assert b.key_end.tolist() == [27, 34, 41, 48]


@pytest.mark.acceptance("R06")
def test_donor_universe_and_frozen_labels():
    e = engine(donors={"T": 0, "D1": 2})
    r = e.retrieve("T", 59, Lc=7)
    assert {c["donor_id"] for c in r["candidates"]} == {"D1"}
    assert {c["donor_group"] for c in r["candidates"]} == {2}  # D1's inferred age group would be 0.
    control = e.retrieve("T", 59, Lc=7, method="all")
    assert r["library_version"] == control["library_version"]
    assert r["scored_candidate_count"] == control["scored_candidate_count"]
    with pytest.raises(DataError, match="training cohort"):
        RetrievalEngine("C", e.source, e.bundle, {"D1": 0, "D2": 1}, ["D1"], {})


@pytest.mark.acceptance("R07")
@pytest.mark.parametrize("Lc", [1, 3, 7, 14])
def test_hidden_attributes_profile_and_perfect_match(Lc):
    # Target has a perfect matching hidden K/V and visible positive records, but is never a cross donor.
    records = [event("T", t, 10, age="2") for t in range(20)]
    records += [event("T", t, 10, age="U") for t in range(59-Lc+1, 60)]
    records += [event("D1", t, 11) for t in range(60)]
    e = engine(records, donors={"T": 0, "D1": 1})
    r = e.retrieve("T", 59, Lc=Lc)
    assert r["history_start"] == 60-Lc and r["Tobs"] == r["ell"] == Lc
    assert r["Q"] == [10.]*Lc
    assert r["profile"]["n"] == Lc and r["profile"]["age"] == "U"
    assert r["attributes"]["attribute_source_step"] >= 60-Lc
    assert r["group_status"] == "GROUP_UNKNOWN"
    assert all(c["donor_id"] == "D1" and c["distance"] > 0 for c in r["candidates"])
    # A window with no raw event cannot restore registry attributes.
    r2 = e.retrieve("T", 30, Lc=Lc)
    if Lc <= 7:
        assert r2["profile"]["age"] == r2["profile"]["gender"] == "U"
    # Even in the SELF route, a long artificial window excludes all earlier target K/V.
    r3 = e.retrieve("T", 59, Lc=30)
    assert r3["retrieval_path"] == "SELF" and r3["candidate_count"] == 0
    assert r3["history_start"] == 30 and not r3["reference_available"]


@pytest.mark.acceptance("R07")
def test_hidden_change_invariance_and_target_gap_block():
    records = [event("T", 0, 999, age="6"), event("T", 59, 10, age="U"), event("D1", 0, 20)]
    e = engine(records, donors={"D1": 0})
    records[0] = event("T", 0, 1, age="0")
    e2 = engine(records, donors={"D1": 0})
    a, b = e.retrieve("T", 59, Lc=3), e2.retrieve("T", 59, Lc=3)
    for key in ("Q", "profile", "attributes", "R", "group_status", "history_start"):
        assert a[key] == b[key]
    e.source.panels["T"] = e.source.panels["T"].loc[lambda x: x.step.ne(58)]
    with pytest.raises(DataError, match="missing or duplicate"):
        e.retrieve("T", 59, Lc=3)


@pytest.mark.acceptance("R08", "R09")
def test_ties_limits_and_self_unlimited():
    records = [event(c, t) for c in ("T", "D2", "D10") for t in range(180)]
    e = engine(records, donors={"D2": 0, "D10": 0})
    a = e.retrieve("T", 59, Lc=7, method="all")
    assert [(c["donor_id"], c["key_end"]) for c in a["candidates"]] == [("D10", 6), ("D10", 20), ("D2", 6), ("D2", 20)]
    assert a == e.retrieve("T", 59, Lc=7, method="all")
    assert [c["weight"] for c in a["candidates"]] == [.25]*4
    own = e.retrieve("T", 112)
    assert own["candidate_count"] == 5 and {c["donor_id"] for c in own["candidates"]} == {"T"}
    assert own["candidates"][1]["key_start"] <= own["candidates"][0]["value_end"]


@pytest.mark.acceptance("R10")
def test_expansion_reselects_without_locking_same_group():
    records = [event(c, t, 20 if c == "D1" else 10) for c in ("T", "D1", "D2", "D3") for t in range(60)]
    e = engine(records, donors={"D1": 0, "D2": 1, "D3": 1}, params=RetrievalParams(m=3))
    r = e.retrieve("T", 59, Lc=7)
    assert r["group_status"] == "FULL" and r["same_group_accepted_count"] == 2
    assert r["selection_status"] == "EXPANDED_ALL" and r["expanded_all"]
    assert all(c["donor_id"] != "D1" for c in r["candidates"])
    assert r["candidate_ids"] == e.retrieve("T", 59, Lc=7, method="all")["candidate_ids"]


@pytest.mark.acceptance("R11", "R12")
def test_actual_toy_and_empty():
    e = toy_engine()
    r = e.retrieve("T", 3)
    assert r["Q"] == [30, 20] and r["group_status"] == "AGE_ONLY"
    assert [c["distance"] for c in r["candidates"]] == [0, 0]
    assert [c["weight"] for c in r["candidates"]] == [.5, .5]
    assert r["R"] == read_json("examples/toy_walkthrough.json")["expected_reference"]
    assert r["candidate_count"] == len(set(r["candidate_ids"])) == 2
    np.testing.assert_allclose(r["r_log"], np.log1p([20, 30]))
    wrong = np.expm1(np.log1p([[10, 20], [30, 40]]).mean(axis=0))
    assert not np.allclose(r["R"], wrong)
    empty = e.retrieve("T", 0)
    assert empty["R"] == empty["r_log"] == [0, 0]
    assert empty["reference_available"] == empty["r_mask"] == empty["candidate_count"] == 0


@pytest.mark.acceptance("R13")
@pytest.mark.parametrize("tau", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_tau(tau):
    with pytest.raises(DataError):
        RetrievalParams(tau=tau)


@pytest.mark.acceptance("R13")
def test_stable_large_distances():
    weights = reference_weights([1e6, 1e6+1, 1e6+1e3], 1)
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(weights[:2], [1/(1+np.exp(-1)), 1/(1+np.exp(1))])


@pytest.mark.acceptance("R14")
def test_cache_identity_reload_and_corruption(tmp_path):
    cache = RetrievalCache(tmp_path / "cache")
    e = engine(cache=cache)
    a = e.retrieve("T", 149, Lc=7, partition="test")
    reloaded = engine(cache=cache).retrieve("T", 149, Lc=7, partition="test")
    assert reloaded.pop("cache_hit") is True
    expected = dict(a)
    expected.pop("cache_hit")
    assert reloaded == expected
    # A new method has no query cache yet; this fresh engine must load the saved date index.
    from_disk = engine(cache=cache)
    disk_result = from_disk.retrieve("T", 149, Lc=7, partition="test", method="all")
    assert not disk_result["cache_hit"] and len(from_disk._banks) == 1
    assert disk_result["candidate_ids"] == a["candidate_ids"]
    assert disk_result["R"] == a["R"]
    requests = [dict(Lc=3), dict(Lc=None), dict(Lc=7, method="all"), dict(Lc=7, library_mode="rolling")]
    variants = [e.retrieve("T", 149, partition="test", **request) for request in requests]
    for change in (dict(stride=1), dict(m=3), dict(tau=2), dict(block_size=1)):
        variants.append(engine(cache=cache, params=replace(e.params, **change)).retrieve("T", 149, Lc=7, partition="test"))
    for version in ({"data_hash": "changed"}, {"cluster_version": "changed"}, {"config_version": "changed"}):
        variants.append(engine(cache=cache, versions=version).retrieve("T", 149, Lc=7, partition="test"))
    variants.append(engine(cache=cache, experiment="C").retrieve("T", 149, Lc=7, partition="test"))
    keys = [a["cache_key"]] + [v["cache_key"] for v in variants]
    assert len(set(keys)) == len(keys)
    assert [v["cache_hit"] for v in variants] == [False, False, True] + [False]*(len(variants)-3)
    rolling = variants[3]
    assert rolling["candidate_cutoff"] == 149 and a["candidate_cutoff"] == 119
    assert rolling["library_version"] != a["library_version"]
    assert rolling["scored_candidate_count"] > a["scored_candidate_count"]
    path = cache.root / "queries" / (a["cache_key"] + ".json")
    path.write_text(path.read_text(encoding="utf-8").replace('"candidate_count": 5', '"candidate_count": 99'), encoding="utf-8")
    with pytest.raises(DataError, match="Corrupt retrieval cache"):
        e.retrieve("T", 149, Lc=7, partition="test")
    bank_path = cache.root / "libraries" / (a["library_version"] + ".npz")
    bank_path.write_bytes(b"deliberately corrupt test index")
    with pytest.raises(DataError, match="Corrupt candidate index"):
        engine(cache=cache).retrieve("D2", 149, Lc=7, partition="test")


@pytest.mark.acceptance("R15")
@pytest.mark.parametrize("block_size", [1, 2, 17, 2048])
def test_blocked_against_independent_enumeration(block_size):
    rng = np.random.default_rng(42)
    records = [event(c, t, int(rng.integers(0, 30))) for c in ("T", "D1", "D2", "D3") for t in range(60)]
    e = engine(records, params=RetrievalParams(block_size=block_size))
    r = e.retrieve("T", 59, Lc=7, method="all")
    oracle = []
    q = np.asarray(r["Q"])
    for donor in ("D1", "D2", "D3"):
        x = e.store.amounts[donor]
        for end in range(6, 53, 7):
            d = float(sum((np.log1p(q[k])-np.log1p(x[end-6+k]))**2 for k in range(7))/7)
            oracle.append((d, donor, end, x[end+1:end+8].tolist()))
    oracle.sort(key=lambda a: a[:3])
    selected = []
    for candidate in oracle:
        _, donor, end, _ = candidate
        same = [c for c in selected if c[1] == donor]
        if len(same) < 2 and all(end-6 > c[2]+7 or c[2]-6 > end+7 for c in same):
            selected.append(candidate)
        if len(selected) == 5:
            break
    assert [(c["donor_id"], c["key_end"]) for c in r["candidates"]] == [(c[1], c[2]) for c in selected]
    assert r["candidate_ids"] == [f"TEST:{r['library_version']}:{c[1]}:{c[2]-6}:{c[2]}:{c[2]+7}" for c in selected]
    np.testing.assert_allclose([c["distance"] for c in r["candidates"]], [c[0] for c in selected], rtol=1e-12)
    w = np.exp(-np.array([c[0] for c in selected]))
    w /= w.sum()
    np.testing.assert_allclose(r["R"], np.sum(w[:, None]*np.array([c[3] for c in selected]), axis=0), rtol=1e-12)
    assert r["max_distance_block_rows"] <= block_size
    assert r["scored_candidate_count"] == len(oracle)


@pytest.mark.acceptance("R06", "R07")
@pytest.mark.parametrize("age,gender,status", [("2", "F", "AGE_ONLY"), ("U", "F", "GROUP_UNKNOWN"), ("2", "E", "NON_PERSON")])
def test_non_full_uses_all_pool(age, gender, status):
    records = [event("T", 59, 10, age=age, gender=gender)] + [event("D1", 0)]
    e = engine(records, donors={"D1": 2})
    r = e.retrieve("T", 59, Lc=1)
    assert r["group_status"] == status and r["selection_status"] == "ALL_DONORS"
    assert status in r["fallback_reason"] and not r["expanded_all"]
