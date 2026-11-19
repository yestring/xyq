"""Strict lineage checks before loading locally generated PyTorch weights."""

import importlib.metadata
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from .cluster_io import fingerprint
from .data import DataError, sha256_file
from .data_stage import dump_json
from .forecast_model import ConsumptionMLP
from .forecast_runtime import io_path
from .retrieval_io import read_json


def model_code_version():
    names = ["forecast_model.py","forecast_data.py","forecast_cache.py","forecast_metrics.py",
             "forecast_training.py","forecast_checkpoint.py","forecast_predict.py",
             "forecast_runtime.py","forecast_runner.py","forecast_train_cli.py"]
    return fingerprint({name:sha256_file(Path("src/banksim")/name) for name in names})


def model_lineage(engine):
    bundle = engine.bundle
    return {"spec_version":"1.0.0","experiment":engine.experiment,"versions":engine.versions,
            "defaults_hash":sha256_file("config/defaults.json"),"model_code_version":model_code_version(),
            "defaults":read_json("config/defaults.json"),
            "architecture":{"history":[56,64],"reference":[8,32],"joint":[96,64,7],
                            "linear_bias":True,"hidden_activation":"ReLU","output_activation":"Softplus","dtype":"float32"},
            "split_ids":read_json(f"artifacts/splits/{engine.experiment}.json")["ids"],
            "a":bundle.amount_scale["a"],"amount_scale":bundle.amount_scale,
            "preprocessing":bundle.transform.to_dict(),"centers":bundle.model.cluster_centers_.tolist(),
            "group_map":bundle.group_map,"fit_cutoff":bundle.metadata["fit_cutoff"],
            "retrieval_parameters":asdict(engine.params),
            "packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","pyarrow","scikit-learn")},
            "torch_version":str(torch.__version__),"rule_version":bundle.metadata["rule_version"]}


def atomic_bytes(path, contents):
    path = io_path(path)
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,path)
    if os.name != "nt":
        fd = os.open(path.parent,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _encode(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")


def committed_manifest(directory):
    """Only committed manifests are authoritative; orphan generations are ignored.

    Reject a broken committed generation and use the previous complete commit.
    Compatibility failures are handled AFTER this function, never by rollback.
    """
    directory = io_path(directory)
    errors = []
    for filename in ("checkpoint_manifest.json","checkpoint_previous.json"):
        path = directory/filename
        if not path.exists():
            continue
        try:
            envelope = read_json(path)
            manifest = envelope["manifest"]
            if envelope["sha256"] != fingerprint(manifest):
                raise ValueError("manifest checksum")
            for pointer in (manifest["last"],manifest.get("best")):
                if pointer is None:
                    continue
                for suffix in ("pt","json"):
                    file = directory/"checkpoints"/(pointer["generation"]+"."+suffix)
                    if sha256_file(file) != pointer[suffix+"_sha256"]:
                        raise ValueError("generation checksum")
            if errors:
                print(json.dumps({"event":"checkpoint_rollback","errors":errors,"using":filename}),flush=True)
            return manifest
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        raise DataError("CHECKPOINT_INVALID", "No complete committed checkpoint", errors=errors)
    return None


def publish_checkpoint(directory, payload, metadata, *, improved=False, fault=None):
    """Caller holds the run writer lock. One commit publishes last AND best.

    Files are immutable generations; a flushed atomic manifest is the commit
    point. Retain the previous commit and every generation referenced by either
    manifest. fault is an internal test hook, never a CLI training option.
    """
    directory = io_path(directory)
    generations = directory/"checkpoints"
    generations.mkdir(parents=True,exist_ok=True)
    previous = committed_manifest(directory)
    generation = uuid.uuid4().hex
    weight_path = generations/(generation+".pt")
    with weight_path.open("xb") as stream:
        torch.save(payload,stream)
        stream.flush()
        os.fsync(stream.fileno())
    if fault: fault("weights_written")
    meta = {**metadata,"weight_sha256":sha256_file(weight_path),"checkpoint_format":2}
    atomic_bytes(generations/(generation+".json"),_encode(meta))
    if fault: fault("metadata_written")
    pointer = {"generation":generation,"pt_sha256":meta["weight_sha256"],
               "json_sha256":sha256_file(generations/(generation+".json"))}
    manifest = {"format":2,"sequence":1 if previous is None else previous["sequence"]+1,
                "last":pointer,"best":pointer if improved else previous.get("best") if previous else None}
    if previous:
        atomic_bytes(directory/"checkpoint_previous.json",_encode({"manifest":previous,"sha256":fingerprint(previous)}))
    if fault: fault("before_commit")
    atomic_bytes(directory/"checkpoint_manifest.json",_encode({"manifest":manifest,"sha256":fingerprint(manifest)}))
    if fault: fault("after_commit")
    # Bounded storage, retaining both committed recovery choices and their best.
    keep = {p["generation"] for m in (manifest,previous) if m for k in ("last","best") if (p:=m.get(k))}
    for file in generations.iterdir():
        if file.suffix in (".pt",".json") and file.stem not in keep:
            file.unlink()
    return generation


def checkpoint_files(directory, name="best"):
    directory = io_path(directory)
    if name not in ("best","last"):
        raise DataError("INVALID_VALUE", "Checkpoint must be best or last")
    manifest = committed_manifest(directory)
    if manifest:
        pointer = manifest.get(name)
        if pointer is None:
            raise DataError("MODEL_NOT_TRAINED", "No validated best model yet")
        base = directory/"checkpoints"/pointer["generation"]
        return base.with_suffix(".pt"),base.with_suffix(".json")
    # Old artifacts remain readable only if their actual lineage is compatible.
    weight,meta = directory/f"{name}.pt",directory/f"{name}.json"
    if not weight.exists() or not meta.exists():
        raise DataError("MODEL_NOT_TRAINED", "No matching trained weights",path=directory.as_posix())
    return weight,meta


def load_checkpoint(directory, engine, *, kind="full", variant="retrieval", method="same_group", name="best", device="cpu"):
    directory = io_path(directory)
    weights,meta = checkpoint_files(directory,name)
    metadata = read_json(meta)
    if (metadata["kind"] != kind or metadata["variant"] != variant or metadata["method"] != method
            or fingerprint(metadata["lineage"]) != fingerprint(model_lineage(engine))):
        raise DataError("CONFIG_MISMATCH", "Checkpoint kind/configuration/data/fit/retrieval/model version differs")
    if sha256_file(weights) != metadata["weight_sha256"]:
        raise DataError("CONFIG_MISMATCH", "Checkpoint bytes differ")
    for entry in metadata["manifests"]:
        if sha256_file(directory/entry["file"]) != entry["sha256"]:
            raise DataError("CONFIG_MISMATCH", "Training or validation selection manifest changed")
    payload = torch.load(weights,map_location="cpu",weights_only=True)
    model = ConsumptionMLP()
    model.load_state_dict(payload["model"])
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise DataError("CONFIG_MISMATCH", "Checkpoint contains nonfinite parameters")
    model.to(device)
    model.eval()
    return model,metadata,payload
