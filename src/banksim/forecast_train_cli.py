"""Explicit new/resume/best-inference entry points; never promotes smoke to full."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .data import DataError
from .data_stage import dump_json
from .forecast_data import LegalSamples
from .forecast_model import resolve_device
from .forecast_predict import predict_customer
from .forecast_stage import setup, smoke_catalogs
from .forecast_training import RunConfig, train_model
from .retrieval_io import read_json


def selected_catalogs(engine, profile="default"):
    catalogs = {engine.experiment:{p:LegalSamples(engine,p) for p in ("train","validation")}}
    small = smoke_catalogs(catalogs)[engine.experiment]
    if profile == "resume-validation":
        origins = list(range(59,71))
        full = catalogs[engine.experiment]["train"]
        ids = [c for c,ts in full.customer_origins.items() if set(origins) <= set(ts)][:4]
        small["train"] = full.subset(ids,origins)
    return small


def saved_catalogs(engine, directory):
    return {p:LegalSamples(engine,p,pd.read_parquet(Path(directory)/f"{p}_selection.parquet"))
            for p in ("train","validation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("new","resume","infer-best"))
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--experiment",choices=("O","C"),required=True)
    parser.add_argument("--kind",choices=("smoke","full"),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--variant",choices=("retrieval","no_retrieval"),default="retrieval")
    parser.add_argument("--method",choices=("same_group","all"),default="same_group")
    parser.add_argument("--device",default="cpu",help="cpu, cuda, or a visible CUDA index such as cuda:0")
    parser.add_argument("--smoke-profile",choices=("default","resume-validation"),default="default")
    parser.add_argument("--checkpoint-every-batches",type=int,default=25)
    parser.add_argument("--progress-seconds",type=float,default=30)
    parser.add_argument("--stop-after-steps",type=int,help="Smoke-only test stop; does not change the 3-epoch plan")
    parser.add_argument("--stop-after-epoch",type=int,help="Smoke-only test stop; does not change the 3-epoch plan")
    parser.add_argument("--confirm-full-training",action="store_true")
    args = parser.parse_args()
    if args.kind == "full" and args.action != "infer-best" and not args.confirm_full_training:
        parser.error("Full training requires separate authorization and --confirm-full-training")
    if not args.run_dir.resolve().is_relative_to((Path.cwd()/"artifacts/stage4").resolve()):
        parser.error("Run directory must be inside artifacts/stage4")
    try:
        device = resolve_device(args.device)
    except DataError as exc:
        parser.error(f"{exc.code}: {exc}")
    with threadpool_limits(limits=2):
        engine = setup()[args.experiment]
        config = RunConfig(kind=args.kind,seed=args.seed,variant=args.variant,method=args.method,
                           batch_size=16 if args.kind == "smoke" else 256,max_epochs=3 if args.kind == "smoke" else 100)
        if args.action in ("resume","infer-best"):
            meta = read_json(args.run_dir/"config.json")
            if meta["config"] != asdict(config) or meta["lineage"]["experiment"] != args.experiment:
                raise DataError("CONFIG_MISMATCH", "Requested run type/experiment/config does not match saved run")
            parts = saved_catalogs(engine,args.run_dir)
        elif args.kind == "smoke":
            parts = selected_catalogs(engine,args.smoke_profile)
        else:
            parts = {p:LegalSamples(engine,p) for p in ("train","validation")}
        if args.action == "infer-best":
            key = parts["validation"].key_at(0)
            result = predict_customer(engine,args.run_dir,key.customer,key.t,Lc=key.Lc,partition="validation",
                                      kind=args.kind,variant=args.variant,method=args.method,admission=parts["validation"],
                                      device=device)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return
        result = train_model(parts["train"],parts["validation"],args.run_dir,config,args.run_dir/"cache",
                             resume=args.action == "resume",checkpoint_every_batches=args.checkpoint_every_batches,
                             progress_seconds=args.progress_seconds,stop_after_steps=args.stop_after_steps,
                             stop_after_epoch=args.stop_after_epoch,device=device)
        print(json.dumps({"status":result.get("status","COMPLETE_SMOKE" if args.kind == "smoke" else "COMPLETE"),
                          "run_dir":args.run_dir.as_posix(),"epochs":result.get("epochs_run"),"device":str(device),
                          "steps":result["completed_steps"]}),flush=True)


if __name__ == "__main__":
    main()
