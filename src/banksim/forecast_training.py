"""Admitted-sample training, customer-balanced validation and best restoration."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .cluster_io import fingerprint
from .data import DataError, sha256_file
from .data_stage import dump_json
from .forecast_cache import CompactReferences
from .forecast_checkpoint import load_checkpoint, model_lineage
from .forecast_data import CustomerSampler
from .forecast_metrics import EarlyStopping, MetricAccumulator, baselines
from .forecast_model import ConsumptionMLP, mask_reference, move_batch, network_inputs, optimizer_for, tensor_inputs, train_step


@dataclass(frozen=True)
class RunConfig:
    kind: str = "smoke"
    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 3
    patience: int = 10
    variant: str = "retrieval"
    method: str = "same_group"

    def __post_init__(self):
        if self.kind not in ("smoke","full") or self.variant not in ("retrieval","no_retrieval") or self.method not in ("same_group","all"):
            raise DataError("INVALID_VALUE", "Invalid run type")
        if self.seed not in (42,43,44) or self.batch_size < 1 or self.max_epochs < 1 or self.patience != 10:
            raise DataError("CONFIG_MISMATCH", "Training protocol differs")
        if self.kind == "full" and (self.batch_size != 256 or self.max_epochs != 100):
            raise DataError("CONFIG_MISMATCH", "Full run requires batch256 and maximum100 epochs")


def prepare_batch(catalog, keys, references, *, no_retrieval=False):
    records,targets = [],[]
    for key in keys:
        _,row = catalog.locate(key)
        x = catalog.engine.store.amounts[key.customer]
        if no_retrieval:
            records.append({"Q":x[key.t-int(row.ell)+1:key.t+1].tolist(),"R":[0.]*7,
                            "r_log":[0.]*7,"reference_available":0,"candidate_count":0})
        else:
            records.append(references.get(key))
        targets.append(catalog.target_cents(key))
    return tensor_inputs(records,targets,catalog.engine.bundle.amount_scale["a"]),records,np.asarray(targets,dtype=np.float64)/100


def evaluate(model, catalog, references, *, variant="retrieval", batch_size=256, collect=False):
    if catalog.partition not in ("validation","test"):
        raise DataError("INVALID_VALUE", "Evaluation needs an admitted evaluation partition")
    catalog.validate()
    # Validate all four strict strata before doing any forward passes.
    lengths = {0} if catalog.experiment == "O" else {1,3,7,14}
    if set(catalog.frame.lc_key) != lengths or catalog.frame.empty:
        raise DataError("VALIDATION_EMPTY", "Required evaluation stratum absent")
    accumulators = {name:MetricAccumulator() for name in ("model","zero","last","q_mean","q_median","retrieval")}
    predictions, timings = [],{"prepare_seconds":0.,"forward_seconds":0.}
    available = 0
    device = next(model.parameters()).device
    model.eval()
    for start in range(0,len(catalog.frame),batch_size):
        keys = [catalog.key_at(i) for i in range(start,min(start+batch_size,len(catalog.frame)))]
        tick = perf_counter()
        batch,records,truth = prepare_batch(catalog,keys,references)
        batch,_ = mask_reference(batch,no_retrieval=variant == "no_retrieval")
        timings["prepare_seconds"] += perf_counter()-tick
        tick = perf_counter()
        with torch.no_grad():
            inputs = move_batch(network_inputs(batch),device)
            output = model(**inputs).cpu().numpy().astype(np.float64)*catalog.engine.bundle.amount_scale["a"]
        timings["forward_seconds"] += perf_counter()-tick
        if not np.isfinite(output).all() or (output < 0).any():
            raise DataError("TRAINING_FAILED", "Nonfinite/negative evaluated prediction")
        for i,key in enumerate(keys):
            variants = {"model":output[i],**baselines(records[i])}
            for name,pred in variants.items():
                accumulators[name].add(key.customer,key.Lc,pred,truth[i])
            available += records[i]["reference_available"]
            if collect:
                predictions.append({"customer":key.customer,"origin_t":key.t,"Lc":key.Lc,"truth":truth[i].tolist(),
                                    "predictions":{k:v.tolist() for k,v in variants.items()},
                                    "reference_available":records[i]["reference_available"]})
    references.flush()
    return {"methods":{name:acc.finish(catalog.experiment) for name,acc in accumulators.items()},
            "samples":len(catalog.frame),"customers":int(catalog.frame.customer.nunique()),
            "available_count":available,"timings":timings,"predictions":predictions}


def train_model(train, validation, output_dir, config, cache_root, **controls):
    """New/resumed runs share one batch-boundary state machine."""
    from .forecast_runner import run_training
    return run_training(train, validation, output_dir, config, cache_root, **controls)
