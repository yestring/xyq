from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from banksim.data import DataError, sha256_file
from banksim.forecast_cache import CompactReferences
from banksim.forecast_checkpoint import load_checkpoint
from banksim.forecast_data import LegalSamples, SampleKey, CustomerSampler
from banksim.forecast_predict import predict_customer
from banksim.forecast_stage import setup, load_catalogs, smoke_catalogs
from banksim.forecast_training import prepare_batch, evaluate, train_model, RunConfig
from banksim.forecast_model import ConsumptionMLP
from banksim.retrieval_io import read_json


@pytest.fixture(scope="module")
def context():
    engines = setup()
    catalogs,_ = load_catalogs(engines)
    return engines,catalogs,smoke_catalogs(catalogs)


@pytest.mark.acceptance("M03","M06","E07","E12")
@pytest.mark.parametrize("experiment", ["O","C"])
def test_saved_sample_admission_and_scale(context,experiment):
    engines,catalogs,small = context
    e = engines[experiment]
    train = catalogs[experiment]["train"]
    assert CustomerSampler(train,42).epoch_size(256) == ((len(train.ordinary)+255)//256)*256
    assert len(train.ordinary) < len(train.frame)
    assert set(train.customers) <= set(train.split["ids"]["train"])
    accepted = read_json(f"artifacts/stage2/full/{experiment}/amount_scale.json")
    assert e.bundle.amount_scale["a"] == accepted["a"] and accepted["fit_end"] == 59
    row = small[experiment]["train"].frame.iloc[[0]].copy()
    broken = row.copy(); broken["target_end"] += 1
    with pytest.raises(DataError):
        LegalSamples(e,"train",broken)
    broken = row.copy(); broken["history_start"] = -1
    with pytest.raises(DataError):
        LegalSamples(e,"train",broken)
    with pytest.raises(DataError):
        train.locate(SampleKey(row.iloc[0].customer,119))
    if experiment == "C":
        broken = row.copy(); broken["customer"] = train.split["ids"]["test"][0]
        with pytest.raises(DataError):
            LegalSamples(e,"train",broken)
        assert not set(e.donor_groups) & (set(train.split["ids"]["validation"]) | set(train.split["ids"]["test"]))
    altered = copy(e); altered.bundle = copy(e.bundle)
    altered.bundle.metadata = {**e.bundle.metadata,"fit_cutoff":119}
    with pytest.raises(DataError,match="fit lineage"):
        LegalSamples(altered,"train",row)


@pytest.mark.acceptance("M01","M04","M07","E08","E12")
@pytest.mark.parametrize("experiment", ["O","C"])
def test_compact_cache_equivalent_reload_and_corrupt(context,experiment,tmp_path):
    _,_,small = context
    catalog = small[experiment]["train"]
    references = CompactReferences(catalog,tmp_path/"cache")
    keys = [catalog.key_at(i) for i in range(min(8,len(catalog.frame)))]
    for key in keys:
        direct = catalog.engine.retrieve(key.customer,key.t,Lc=key.Lc,partition="train")
        cached = references.get(key)
        assert not cached["cache_hit"]
        assert direct["R"] == cached["R"] and direct["r_log"] == cached["r_log"]
        assert [c["candidate_id"] for c in direct["candidates"]] == [c["candidate_id"] for c in cached["trace"]]
        assert [(c["distance"],c["weight"]) for c in direct["candidates"]] == [(c["distance"],c["weight"]) for c in cached["trace"]]
    references.close()
    reload = CompactReferences(catalog,tmp_path/"cache")
    assert all(reload.get(key)["cache_hit"] for key in keys)
    batch,records,truth = prepare_batch(catalog,keys,reload)
    for i,key in enumerate(keys):
        original = catalog.engine.source.panels[key.customer]
        cents = original.loc[original.step.between(key.t+1,key.t+7),"target_cents"].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(truth[i],cents/100)
        np.testing.assert_allclose(batch["y_scaled"][i].numpy(),cents/100/catalog.engine.bundle.amount_scale["a"],rtol=1e-6)
    i,_ = catalog.locate(keys[0]); reload.rows[i]["R"][0] += 1
    with pytest.raises(DataError,match="checksum"):
        reload.get(keys[0])
    reload.close()


@pytest.mark.acceptance("M08","M09","M11","M12","E08","A07")
@pytest.mark.parametrize("experiment", ["O","C"])
def test_actual_smoke_independent_models_best_restore_and_predictions(context,experiment,request):
    engines,_,small = context
    reports = []
    states = []
    validation_keys = []
    for variant in ("retrieval","no_retrieval"):
        directory = Path(request.config.getoption("--forecast-artifacts-root"))/"smoke"/experiment/variant
        report = read_json(directory/"summary.json")
        reports.append(report)
        assert report["kind"] == "smoke" and report["seed"] == 42 and report["epochs_run"] == 3
        assert report["best_restored"] and report["reload_equal"]
        assert np.isfinite([r["scaled_training_mse"] for r in report["curve"]]).all()
        model,metadata,payload = load_checkpoint(directory,engines[experiment],kind="smoke",variant=variant)
        assert payload["epoch"] == report["best_epoch"]
        assert {"optimizer","torch_rng","reference_dropout_rng","sampler_rng","bad_epochs"} <= payload.keys()
        assert metadata["lineage"]["fit_cutoff"] == 59
        assert metadata["lineage"]["a"] == report["a"]
        states.append(payload["model"])
        predictions = report["validation"]["predictions"]
        validation_keys.append([(r["customer"],r["origin_t"],r["Lc"],r["truth"]) for r in predictions])
        assert len(predictions) == (8 if experiment == "O" else 32)
        assert all(np.isfinite(r["predictions"]["model"]).all() and min(r["predictions"]["model"]) >= 0 for r in predictions)
        example = read_json(directory/"prediction_example.json")
        assert example["prediction_status"] == "SMOKE_ONLY" and example["kind"] == "smoke"
        fields = {"customer","origin_t","target_steps","history_start","Tobs","ell","H","predicted_daily",
                  "predicted_total","prediction_status","group_name","group_status","allocation_status",
                  "r_conservative","r_aggressive","retrieval_mode","reference_available","candidate_count",
                  "candidate_ids","fallback_reason","model_version","cluster_version","rule_version","data_hash"}
        assert fields <= example.keys()
        assert example["predicted_total"] == pytest.approx(sum(example["predicted_daily"]),rel=0,abs=1e-12)
    assert validation_keys[0] == validation_keys[1]
    assert reports[0]["model_version"] != reports[1]["model_version"]
    assert any(not torch.equal(states[0][k],states[1][k]) for k in states[0])
    if experiment == "O":
        assert reports[0]["best_epoch"] < reports[0]["epochs_run"]


@pytest.mark.acceptance("M11","E12","A07")
def test_missing_model_and_incompatible_lineage(context,tmp_path,monkeypatch,request):
    engines,_,small = context
    e = engines["O"]
    key = small["O"]["validation"].key_at(0)
    absent = predict_customer(e,tmp_path/"absent",key.customer,key.t,partition="validation")
    assert absent["prediction_status"] == "MODEL_NOT_TRAINED" and absent["predicted_daily"] is None
    directory = Path(request.config.getoption("--forecast-artifacts-root"))/"smoke/O/retrieval"
    with pytest.raises(DataError) as exc:
        load_checkpoint(directory,e,kind="full")
    assert exc.value.code == "CONFIG_MISMATCH"
    for key,value in (("data_hash","different"),("cluster_version","different"),("implementation_version","different")):
        with monkeypatch.context() as m:
            m.setitem(e.versions,key,value)
            with pytest.raises(DataError) as exc:
                load_checkpoint(directory,e,kind="smoke")
            assert exc.value.code == "CONFIG_MISMATCH"
    for params in (replace(e.params,L=14),replace(e.params,m=3)):
        with monkeypatch.context() as m:
            m.setattr(e,"params",params)
            with pytest.raises(DataError):
                load_checkpoint(directory,e,kind="smoke")
    with monkeypatch.context() as m:
        m.setitem(e.bundle.amount_scale,"a",999.)
        with pytest.raises(DataError):
            load_checkpoint(directory,e,kind="smoke")


@pytest.mark.acceptance("M10","E07")
def test_empty_strict_validation_blocks_before_training(context,tmp_path):
    _,_,small = context
    train,val = small["C"]["train"],small["C"]["validation"]
    bad = LegalSamples(val.engine,"validation",val.frame.loc[val.frame.lc_key.ne(14)])
    with pytest.raises(DataError) as exc:
        train_model(train,bad,tmp_path/"must_not_train",RunConfig(),tmp_path/"cache")
    assert exc.value.code == "VALIDATION_EMPTY"
    assert not (tmp_path/"must_not_train/best.pt").exists()


@pytest.mark.acceptance("E08","E12")
def test_resource_measurements_and_original_evidence():
    report = read_json("artifacts/stage4/benchmark/report.json")
    assert report["all_reference_equivalent"] and len(report["queries"]) >= 100
    assert report["memory"]["peak_rss_bytes"] > 0
    for g in report["groups"]:
        assert g["samples"] == 12 and g["timings_seconds"]["compact_hit"]["mean"] > 0
        if g["partition"] == "validation":
            assert g["batch256_train_seconds"] is None  # Held-out labels never used for benchmark gradient steps.
    for path,saved in read_json("artifacts/stage4/baseline.json").items():
        assert sha256_file(path) == saved["sha256"],path
    before = read_json("artifacts/stage4/source-before.json")
    for name in ("candidates.py","retrieval.py","profiles.py","clustering.py","splits.py","data.py","panel.py","allocation.py"):
        assert sha256_file(Path("src/banksim")/name) == before[name]
