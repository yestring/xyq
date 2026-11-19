"""Explicit model availability, smoke identity, and traceable inference."""

import torch

from .allocation import map_allocation
from .data import DataError
from .forecast_checkpoint import load_checkpoint
from .forecast_model import mask_reference, move_batch, network_inputs, tensor_inputs


def predict_customer(engine, directory, customer, t, *, Lc=None, partition="test", kind="full", variant="retrieval", method="same_group", admission=None, device="cpu"):
    base = {"customer":customer,"origin_t":t,"target_steps":list(range(t+1,t+8)),"kind":kind,
            "predicted_daily":None,"predicted_total":None,"spec_version":"1.0.0","data_hash":engine.versions["data_hash"]}
    try:
        model,metadata,_ = load_checkpoint(directory,engine,kind=kind,variant=variant,method=method,device=device)
    except DataError as exc:
        return {**base,"prediction_status":exc.code,"error":exc.as_dict()}
    if admission is not None:
        from .forecast_data import SampleKey
        if admission.partition != partition or admission.experiment != engine.experiment:
            raise DataError("INVALID_VALUE", "Prediction admission partition differs")
        admission.locate(SampleKey(customer,t,Lc))
    r = engine.retrieve(customer,t,Lc=Lc,partition=partition,method=method)
    if r["Tobs"] == 0:
        return {**base,"prediction_status":"NO_HISTORY","group_status":"NO_HISTORY"}
    batch = tensor_inputs([r])
    batch,_ = mask_reference(batch,no_retrieval=variant == "no_retrieval")
    with torch.no_grad():
        inputs = move_batch(network_inputs(batch),next(model.parameters()).device)
        amounts = model(**inputs)[0].cpu().numpy().astype(float)*metadata["lineage"]["a"]
    allocation = map_allocation({"group_name":r["group_name"],"group_status":r["group_status"]})
    return {**base,**allocation,**{key:r[key] for key in ("history_start","Tobs","ell","H","group_name","group_status",
            "reference_available","candidate_count","candidate_ids","fallback_reason")},
            "retrieval_mode":r["retrieval_path"],"retrieval_method":method,"variant":variant,
            "predicted_daily":amounts.tolist(),"predicted_total":float(amounts.sum()),
            "prediction_status":"SMOKE_ONLY" if kind == "smoke" else "OK",
            "model_version":metadata["model_version"],"cluster_version":engine.versions["cluster_version"],
            "rule_version":metadata["lineage"]["rule_version"]}
