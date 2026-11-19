from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from banksim.data import DataError
from banksim.forecast_data import CustomerSampler
from banksim.forecast_metrics import EarlyStopping, MetricAccumulator, baselines
from banksim.forecast_model import (
    ConsumptionMLP,
    mask_reference,
    move_batch,
    optimizer_for,
    resolve_device,
    tensor_inputs,
    train_step,
)
from banksim.forecast_training import RunConfig
from banksim.retrieval_io import read_json
from banksim.retrieval_version import semantic_version
from banksim.profiles import fit_amount_scale


def record(Q=None, R=None, available=1):
    R = [2.,3.,4.,5.,6.,7.,8.] if R is None else R
    return {"Q":[0.,10.] if Q is None else Q,"R":R,"r_log":np.log1p(R).tolist(),"reference_available":available}


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
def test_forecast_training_step_on_requested_device(device_name, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        device = resolve_device(device_name)
    except DataError as err:
        if err.code == "CUDA_UNAVAILABLE":
            pytest.skip("CUDA is unavailable")
        raise
    model = ConsumptionMLP().to(device)
    batch = move_batch(tensor_inputs([record()], [[100] * 7]), device)
    loss = train_step(model, optimizer_for(model), batch)
    assert np.isfinite(loss)
    assert all(param.device == device for param in model.parameters())
    assert all(tensor.device == device for tensor in batch.values())


def test_explicit_cuda_request_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DataError) as err:
        resolve_device("cuda:0")
    assert err.value.code == "CUDA_UNAVAILABLE"


@pytest.mark.acceptance("M01","M02","M04","M07")
def test_tensor_masks_units_and_single_log():
    batch = tensor_inputs([record()],[[1000,3000,0,100,200,300,400]],10)
    assert all(v.dtype == torch.float32 for v in batch.values())
    assert {k:tuple(v.shape) for k,v in batch.items()} == {"q_log":(1,28),"q_mask":(1,28),"r_log":(1,7),"r_mask":(1,1),"y_scaled":(1,7)}
    assert torch.equal(batch["q_mask"][0],torch.tensor([0.]*26+[1.,1.]))
    assert batch["q_log"][0,-2] == 0 and batch["q_log"][0,-1] == pytest.approx(np.log(11))
    np.testing.assert_allclose(batch["r_log"].numpy()[0],np.log1p(record()["R"]),rtol=1e-6)
    assert batch["y_scaled"][0,:2].tolist() == [1.,3.]
    scaled = torch.tensor([2.,3.])
    assert (scaled*10).tolist() == [20.,30.]
    assert ((scaled-batch["y_scaled"][0,:2])**2).mean() == .5


@pytest.mark.acceptance("M03")
def test_amount_scale_daily_warm_training_only_and_fallback():
    panel = pd.DataFrame({"customer":["A","A","B","A","HELDOUT"],"step":[0,1,0,60,0],
                          "observed":[True]*5,"target_cents":[4000,2000,10000,99900,999999]})
    assert fit_amount_scale(panel,["A","B"])["a"] == 40.
    panel.loc[panel.customer.eq("HELDOUT") | panel.step.gt(59),"target_cents"] = 10**9
    assert fit_amount_scale(panel,["A","B"])["a"] == 40.
    panel.loc[panel.customer.isin(["A","B"]),"target_cents"] = 0
    assert fit_amount_scale(panel,["A","B"])["a"] == 1.


@pytest.mark.acceptance("M02","M05","M10")
def test_exact_network_forward_backward_empty_reference():
    model = ConsumptionMLP()
    layers = [m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    assert [(m.in_features,m.out_features) for m in layers] == [(56,64),(8,32),(96,64),(64,7)]
    assert all(m.bias is not None for m in layers)
    assert sum(isinstance(m,torch.nn.Softplus) for m in model.modules()) == 1
    batch = tensor_inputs([record(Q=[0]*28,available=0)],[[0]*7],26.81)
    assert torch.count_nonzero(batch["r_log"]) == torch.count_nonzero(batch["r_mask"]) == 0
    before = {k:v.clone() for k,v in model.state_dict().items()}
    loss = train_step(model,optimizer_for(model),batch)
    assert np.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    out = model(**{k:v for k,v in batch.items() if k != "y_scaled"})
    assert torch.isfinite(out).all() and (out > 0).all()


@pytest.mark.acceptance("M06")
def test_customer_then_origin_sampling_and_epoch_size():
    catalog = SimpleNamespace(partition="train",customers=["A","B"],ordinary=list(range(7)),
                              customer_origins={"A":np.array([59]),"B":np.array([59,60,61,62,63,64])},
                              cold_options={(c,t):np.array([1,3,7,14]) for c,ts in {"A":[59],"B":[59,60,61,62,63,64]}.items() for t in ts})
    s = CustomerSampler(catalog,42)
    assert s.epoch_size(4) == 8
    rows = [s.draw() for _ in range(20000)]
    counts = Counter(r.customer for r in rows)
    assert .48 < counts["A"]/len(rows) < .52
    cold = [r for r in rows if r.Lc is not None]
    assert .48 < len(cold)/len(rows) < .52
    assert all(.23 < v/len(cold) < .27 for v in Counter(r.Lc for r in cold).values())
    b = Counter(r.t for r in rows if r.customer == "B")
    assert all(.14 < v/counts["B"] < .19 for v in b.values())
    again = CustomerSampler(catalog,42)
    assert rows[:20] == [again.draw() for _ in range(20)]
    catalog.cold_options = {}
    empty = CustomerSampler(catalog,42)
    assert all(empty.draw().Lc is None for _ in range(100)) and empty.no_feasible_cold > 0


@pytest.mark.acceptance("M07","M08")
def test_reference_dropout_and_independent_mode():
    batch = tensor_inputs([record() for _ in range(10000)],[[100]*7]*10000,10)
    masked,count = mask_reference(batch,training=True,generator=torch.Generator().manual_seed(42))
    assert 1800 < count < 2200
    hidden = masked["r_mask"][:,0] == 0
    assert torch.count_nonzero(masked["r_log"][hidden]) == 0
    assert torch.equal(masked["y_scaled"],batch["y_scaled"])
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    validation,count = mask_reference(batch,training=False,generator=generator)
    assert count == 0 and torch.equal(generator.get_state(),before)
    assert all(torch.equal(validation[k],v) for k,v in batch.items())
    no,count = mask_reference(batch,no_retrieval=True)
    assert count == 10000 and torch.count_nonzero(no["r_log"]) == torch.count_nonzero(no["r_mask"]) == 0


@pytest.mark.acceptance("M09")
def test_early_stop_ties_and_patience():
    stopper = EarlyStopping(10)
    assert stopper.update(5.,1) == (True,False)
    assert stopper.update(4.,2) == (True,False)
    assert stopper.update(4.,3) == (False,False)
    for epoch in range(4,12):
        assert stopper.update(5.,epoch) == (False,False)
    assert stopper.update(4.,12) == (False,True)
    assert stopper.best_epoch == 2 and stopper.best == 4


@pytest.mark.acceptance("M10")
@pytest.mark.parametrize("failure", ["loss","gradient"])
def test_nonfinite_training_rejected(failure):
    model = ConsumptionMLP()
    batch = tensor_inputs([record()],[[100]*7],10)
    if failure == "loss":
        batch["y_scaled"][:] = float("inf")
    else:
        next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient,float("nan")))
    with pytest.raises(DataError) as exc:
        train_step(model,optimizer_for(model),batch)
    assert exc.value.code == "TRAINING_FAILED"


@pytest.mark.acceptance("E05","E06")
def test_customer_equal_metrics_exact_examples():
    acc = MetricAccumulator()
    for c,p in [("A",0),("A",2),("B",6)]:
        acc.add(c,None,[p],[0])
    m = acc.finish("O")["metrics"]
    assert m == {"MSE":19.,"RMSE":np.sqrt(19.),"MAE":3.5,"TotalAE":3.5}
    acc = MetricAccumulator(); acc.add("T",None,[18,28],[20,30])
    assert acc.finish("O")["metrics"] == {"MSE":4.,"RMSE":2.,"MAE":2.,"TotalAE":4.}


@pytest.mark.acceptance("E07","M10")
def test_strict_equal_length_metrics_and_empty_fail():
    acc = MetricAccumulator()
    for lc in (1,3,7,14):
        acc.add("A",lc,[lc],[0])
        for _ in range(lc):
            acc.add("B",lc,[2*lc],[0])
    result = acc.finish("C")
    assert result["selection_mse"] == np.mean([2.5*lc**2 for lc in (1,3,7,14)])
    del acc.rows[(14,"A")]; del acc.rows[(14,"B")]
    with pytest.raises(DataError) as exc:
        acc.finish("C")
    assert exc.value.code == "VALIDATION_EMPTY"
    with pytest.raises(DataError):
        MetricAccumulator().finish("O")


@pytest.mark.acceptance("E08")
def test_five_baselines_only_query_and_empty_fallback():
    result = baselines(record(Q=[0,10,20],R=[1,2,3,4,5,6,7]))
    assert result["zero"].tolist() == [0]*7
    assert result["last"].tolist() == [20]*7
    assert result["q_mean"].tolist() == result["q_median"].tolist() == [10]*7
    assert result["retrieval"].tolist() == [1,2,3,4,5,6,7]
    assert baselines(record(Q=[0,10,20],available=0))["retrieval"].tolist() == [10]*7


@pytest.mark.acceptance("M12","E12")
def test_seed_contract_and_full_guard():
    states = []
    for seed in (42,43,44):
        torch.manual_seed(seed); states.append(next(ConsumptionMLP().parameters()).detach().clone())
    assert not torch.equal(states[0],states[1]) and not torch.equal(states[1],states[2])
    with pytest.raises(DataError):
        RunConfig(kind="full",batch_size=16,max_epochs=3)
    cfg = RunConfig(kind="full",batch_size=256,max_epochs=100)
    assert cfg.seed == 42 and cfg.patience == 10


@pytest.mark.acceptance("E12")
def test_retrieval_dependency_scoping(tmp_path):
    baseline = read_json("config/retrieval_compatibility.json")
    for name in baseline["dependency_hashes"]:
        (tmp_path/name).write_bytes((Path("src/banksim")/name).read_bytes())
    old = semantic_version(tmp_path)
    assert old == baseline["stage3_implementation_version"]
    (tmp_path/"unrelated_forecaster.py").write_text("new code",encoding="utf-8")
    assert semantic_version(tmp_path) == old
    (tmp_path/"candidates.py").write_text("changed retrieval semantics",encoding="utf-8")
    assert semantic_version(tmp_path) != old
