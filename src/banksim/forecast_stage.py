"""4A smoke/resource CLI; full training requires a separate explicit invocation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .data import DataError, sha256_file
from .data_stage import dump_json
from .forecast_cache import CompactReferences, ROW_DTYPE
from .forecast_data import LegalSamples
from .forecast_model import ConsumptionMLP, mask_reference, optimizer_for, train_step
from .forecast_predict import predict_customer
from .forecast_resources import disk_bytes, process_memory
from .forecast_training import RunConfig, evaluate, prepare_batch, train_model
from .retrieval_cache import RetrievalCache
from .retrieval_io import load_engines, read_json


def setup():
    accepted = read_json("artifacts/stage3/acceptance-final.json")
    if not all(v["status"] == "PASS" for v in accepted["acceptance"].values()):
        raise DataError("CONFIG_MISMATCH", "Stage-three acceptance is incomplete")
    cfg = read_json("config/defaults.json")
    if cfg["network"] != {"history_hidden":64,"reference_hidden":32,"joint_hidden":64,"output_activation":"Softplus",
                          "dtype":"float32","output_scale":"warm_train_positive_daily_median","inverse_expm1":False}:
        raise DataError("CONFIG_MISMATCH", "Network protocol differs")
    expected = {"optimizer":"Adam","lr":.001,"betas":[.9,.999],"eps":1e-8,"weight_decay":0,"batch_size":256,
                "max_epochs":100,"patience":10,"min_delta":0,"loss":"MSE_on_y_divided_by_a","truncate_probability":.5,
                "reference_dropout_probability":.2,"model_seeds":[42,43,44]}
    if cfg["training"] != expected:
        raise DataError("CONFIG_MISMATCH", "Training protocol differs")
    torch.set_num_threads(2)
    return load_engines(cache_root=None)


def load_catalogs(engines):
    started = perf_counter()
    catalogs = {e:{part:LegalSamples(engine,part) for part in ("train","validation")} for e,engine in engines.items()}
    return catalogs,perf_counter()-started


def smoke_catalogs(catalogs):
    selected = {}
    for experiment,parts in catalogs.items():
        train = parts["train"]
        # Explicit small legal cohort: first four IDs with all requested ordinary origins.
        origins = [59,62,89]
        customers = [c for c,ts in train.customer_origins.items() if set(origins) <= set(ts)][:4]
        val = parts["validation"]
        valid = val.frame.loc[val.frame.origin_t.isin([119,126])]
        needed = 2 if experiment == "O" else 8
        eligible = valid.groupby("customer").size()
        validation_customers = sorted(eligible[eligible == needed].index)[:4]
        selected[experiment] = {"train":train.subset(customers,origins),
                                "validation":val.subset(validation_customers,[119,126])}
    return selected


def smoke(catalogs, root, experiments):
    selected = smoke_catalogs(catalogs)
    summary = []
    for experiment in experiments:
        parts = selected[experiment]
        for variant in ("retrieval","no_retrieval"):
            output = root/"smoke"/experiment/variant
            result = train_model(parts["train"],parts["validation"],output,RunConfig(variant=variant),root/"smoke/cache")
            key = parts["validation"].key_at(0)
            prediction = predict_customer(parts["train"].engine,output,key.customer,key.t,Lc=key.Lc,
                                          partition="validation",kind="smoke",variant=variant,admission=parts["validation"])
            dump_json(output/"prediction_example.json",prediction)
            summary.append(result)
    dump_json(root/"smoke/summary.json",{"kind":"smoke_not_full_performance","runs":summary,
              "selection":"First four legal customers; train origins59/62/89, validation119/126; C all four Lc; independent retrieval/no_retrieval models",
              "memory":process_memory()})
    return summary


def benchmark(catalogs, root, load_seconds):
    destination = root/"benchmark"
    destination.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    groups,details = [],[]
    for experiment,parts in catalogs.items():
        for partition,catalog in parts.items():
            f = catalog.frame
            for lc,rows in f.groupby("lc_key",sort=True):
                # Ordinary mode is measured on the requested SELF path.
                if lc == 0:
                    rows = rows.loc[rows.Tobs.gt(14)]
                positions = rng.choice(rows.index.to_numpy(),size=min(12,len(rows)),replace=False)
                chosen = LegalSamples(catalog.engine,partition,f.loc[positions].sort_values(["customer","origin_t"]))
                label = f"{experiment}_{partition}_{'SELF' if lc == 0 else 'CROSS'}_Lc{lc}"
                cache = CompactReferences(chosen,destination/"compact")
                original_cache = chosen.engine.cache
                chosen.engine.cache = None
                times = {"data_prepare":[],"retrieval_compute":[],"compact_first":[],"compact_hit":[],"json_first":[],"json_hit":[]}
                records = []
                for i in range(len(chosen.frame)):
                    key = chosen.key_at(i)
                    tick = perf_counter()
                    prepare_batch(chosen,[key],None,no_retrieval=True)
                    times["data_prepare"].append(perf_counter()-tick)
                    tick = perf_counter()
                    direct = chosen.engine.retrieve(key.customer,key.t,Lc=key.Lc,partition=partition)
                    times["retrieval_compute"].append(perf_counter()-tick)
                    tick = perf_counter(); cached = cache.get(key)
                    times["compact_first"].append(perf_counter()-tick)
                    for field in ("R","r_log","candidate_count","reference_available"):
                        assert direct[field] == cached[field]
                    for full,compact in zip(direct["candidates"],cached["trace"]):
                        assert all(full[k] == v for k,v in compact.items())
                    records.append(direct)
                    details.append({"stratum":label,"customer":key.customer,"origin_t":key.t,"Lc":key.Lc,
                                    "equivalent":True,"candidate_count":direct["candidate_count"]})
                cache.flush()
                for i in range(len(chosen.frame)):
                    tick = perf_counter(); assert cache.get(chosen.key_at(i))["cache_hit"]
                    times["compact_hit"].append(perf_counter()-tick)
                diagnostic = RetrievalCache(destination/"diagnostic_json"/label)
                chosen.engine.cache = diagnostic
                for i in range(len(chosen.frame)):
                    key = chosen.key_at(i)
                    tick = perf_counter(); fresh = chosen.engine.retrieve(key.customer,key.t,Lc=key.Lc,partition=partition)
                    times["json_first"].append(perf_counter()-tick)
                    assert not fresh["cache_hit"]
                    tick = perf_counter(); hit = chosen.engine.retrieve(key.customer,key.t,Lc=key.Lc,partition=partition)
                    times["json_hit"].append(perf_counter()-tick)
                    assert hit["cache_hit"] and hit["R"] == records[i]["R"]
                chosen.engine.cache = original_cache
                keys = [chosen.key_at(i) for i in range(len(chosen.frame))]
                # Batch256 compute benchmark repeats admitted measurements solely to time compute.
                repeated = (keys*((256+len(keys)-1)//len(keys)))[:256]
                tick = perf_counter(); batch,_,_ = prepare_batch(chosen,repeated,cache)
                batch_prepare = perf_counter()-tick
                steps = []
                if partition == "train":
                    model = ConsumptionMLP(); optimizer = optimizer_for(model)
                    train_step(model,optimizer,batch)  # optimizer allocation warmup excluded from steady timing
                    for _ in range(10):
                        tick = perf_counter(); train_step(model,optimizer,batch); steps.append(perf_counter()-tick)
                entry = {"stratum":label,"experiment":experiment,"partition":partition,"Lc":int(lc),"samples":len(keys),
                         "timings_seconds":{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"p95":float(np.quantile(v,.95))} for k,v in times.items()},
                         "batch256_prepare_seconds":batch_prepare,"batch256_train_seconds":float(np.mean(steps)) if steps else None,
                         "compact_allocated_bytes":disk_bytes(cache.root),"compact_row_bytes":ROW_DTYPE.itemsize,
                         "json_query_bytes":disk_bytes(diagnostic.root/"queries"),"json_library_bytes":disk_bytes(diagnostic.root/"libraries"),
                         "memory":process_memory()}
                if partition == "validation":
                    # C evaluate needs all four lengths, measured below on the smoke validation cohort.
                    entry["validation_timing_in_separate_smoke_subset"] = True
                groups.append(entry)
                print(f"benchmark {label}: n={len(keys)} compute={entry['timings_seconds']['retrieval_compute']['mean']:.4f}s compact_hit={entry['timings_seconds']['compact_hit']['mean']:.6f}s",flush=True)
    validation_times = {}
    for exp,parts in smoke_catalogs(catalogs).items():
        catalog = parts["validation"]
        cache = CompactReferences(catalog,destination/"validation_compact")
        model = ConsumptionMLP()
        tick = perf_counter(); first = evaluate(model,catalog,cache,batch_size=256)
        first_seconds = perf_counter()-tick
        tick = perf_counter(); hit = evaluate(model,catalog,cache,batch_size=256)
        validation_times[exp] = {"samples":len(catalog.frame),"first_seconds":first_seconds,"hit_seconds":perf_counter()-tick,
                                 "first_breakdown":first["timings"],"hit_breakdown":hit["timings"]}
    estimates = estimate(catalogs,groups,validation_times)
    report = {"kind":"measured_stratified_small_batch_and_explicit_extrapolation","seed":42,
              "sampling":"12 rows without replacement from each experiment/partition/Lc; ordinary restricted to SELF; NumPy default_rng(42), sequential groups",
              "load_and_admission_seconds":load_seconds,"groups":groups,"queries":details,"validation":validation_times,
              "memory":process_memory(),"all_reference_equivalent":all(d["equivalent"] for d in details),"estimates":estimates}
    dump_json(destination/"report.json",report)
    return report


def estimate(catalogs, groups, validation):
    estimates = []
    for exp,parts in catalogs.items():
        train,val = parts["train"],parts["validation"]
        rounded = ((len(train.ordinary)+255)//256)*256
        measured = {(g["partition"],g["Lc"]):g for g in groups if g["experiment"] == exp}
        cold_fill, cold_validation_fill = 0.,0.
        for part,catalog in parts.items():
            for lc,rows in catalog.frame.groupby("lc_key"):
                # Direct first computation includes bank construction; compact_first was measured
                # immediately after the direct call and may benefit from a warm date-index LRU.
                latency = measured[(part,int(lc))]["timings_seconds"]["retrieval_compute"]["mean"]
                contribution = len(rows)*latency
                if int(lc) == 0:
                    short = int(rows.Tobs.le(14).sum())
                    cross_latency = float(np.mean([measured[("train",k)]["timings_seconds"]["retrieval_compute"]["mean"] for k in (1,3,7,14)]))
                    contribution += short*(cross_latency-latency)
                cold_fill += contribution
                if part == "validation":
                    cold_validation_fill += contribution
        probabilities = {0:.5,1:.125,3:.125,7:.125,14:.125}
        hit_per_row = sum(prob*measured[("train",lc)]["batch256_prepare_seconds"]/256 for lc,prob in probabilities.items())
        step = sum(prob*measured[("train",lc)]["batch256_train_seconds"] for lc,prob in probabilities.items())
        first_query_mean = sum(prob*measured[("train",lc)]["timings_seconds"]["retrieval_compute"]["mean"] for lc,prob in probabilities.items())
        val_hit = validation[exp]["hit_seconds"]/validation[exp]["samples"]*len(val.frame)
        steady = rounded*hit_per_row+(rounded/256)*step+val_hit
        nrows = len(train.frame)+len(val.frame)
        diagnostic_bytes = float(np.mean([g["json_query_bytes"]/g["samples"] for g in groups if g["experiment"] == exp]))
        estimates.append({"experiment":exp,"ordinary_training_pairs":len(train.ordinary),"epoch_draws":rounded,
                          "batches_per_epoch":rounded//256,"training_mode_rows":len(train.frame),"validation_samples":len(val.frame),
                          "training_customers":len(train.customers),"validation_customers":int(val.frame.customer.nunique()),
                          "max_epochs":100,"cache_rows_upper_bound_train_validation":nrows,
                          "compact_payload_bytes_upper_bound":nrows*ROW_DTYPE.itemsize,
                          "json_query_bytes_estimate":nrows*diagnostic_bytes,"json_file_count_upper_bound":nrows,
                          "all_unique_reference_fill_seconds_estimate":cold_fill,
                          "steady_epoch_seconds_estimate":steady,"max100_epochs_seconds_estimate":cold_fill+100*steady,
                          "first_epoch_cold_scenario_seconds_estimate":rounded*first_query_mean+cold_validation_fill+(rounded/256)*step,
                          "assumptions":"One retrieval model/method per experiment, CPU2 threads, same machine load; full unique references upper bound filled once using first direct retrieval latency (not warm-bank compact-first latency); all later accesses compact. Training mixes ordinary0.5 and each Lc0.125; small feasibility deviations ignored. Short ordinary cache-fill cost approximated by mean cross latency. Validation latency extrapolated from measured subset. Disk excludes filesystem overhead/checkpoints/logs; no early-stop benefit assumed for100 epochs. Estimates are scenarios, not measured full runtime or guaranteed limits."})
    return estimates


def refine_estimates(catalogs, root):
    """Larger validation timing only; no fitting, checkpoint selection or test-set use."""
    destination = root/"benchmark"
    report = read_json(destination/"report.json")
    dump_json(destination/"report-before-estimation-refinement.json",report)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    validation_times = {}
    for exp,parts in catalogs.items():
        original = parts["validation"]
        selected = []
        for _,rows in original.frame.groupby("lc_key",sort=True):
            size = 512 if exp == "O" else 128
            selected.extend(rng.choice(rows.index.to_numpy(),size=size,replace=False).tolist())
        catalog = LegalSamples(original.engine,"validation",original.frame.loc[selected].sort_values(["customer","origin_t","lc_key"]))
        catalog.frame.to_parquet(destination/f"validation_timing_{exp}.parquet",index=False)
        cache = CompactReferences(catalog,destination/"validation_512_compact")
        model = ConsumptionMLP()
        tick = perf_counter(); first = evaluate(model,catalog,cache,batch_size=256)
        first_seconds = perf_counter()-tick
        repeated = []
        for _ in range(3):
            tick = perf_counter(); hit = evaluate(model,catalog,cache,batch_size=256)
            repeated.append(perf_counter()-tick)
        validation_times[exp] = {"samples":len(catalog.frame),"customers":int(catalog.frame.customer.nunique()),
                                "first_seconds":first_seconds,"hit_seconds":float(np.mean(repeated)),"hit_repeats_seconds":repeated,
                                "first_breakdown":first["timings"],"hit_breakdown":hit["timings"]}
        print(f"validation timing {exp}: 512 samples; first={first_seconds:.3f}s warm_mean={np.mean(repeated):.3f}s",flush=True)
    report["validation_initial_small"] = report["validation"]
    report["validation"] = validation_times
    report["validation_timing_sampling"] = "Separate default_rng42: O512 validation rows; C128 per Lc. Untrained timing-only model, no backward pass, no checkpoint, no test data. Warm timing is mean of3 passes."
    report["estimates"] = estimate(catalogs,report["groups"],validation_times)
    report["refinement_memory"] = process_memory()
    dump_json(destination/"report.json",report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("smoke","benchmark","estimate","full"),required=True)
    parser.add_argument("--experiment",choices=("O","C","both"),default="both")
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--variant",choices=("retrieval","no_retrieval"),default="retrieval")
    parser.add_argument("--method",choices=("same_group","all"),default="same_group")
    parser.add_argument("--confirm-full-training",action="store_true")
    parser.add_argument("--output",default="artifacts/stage4")
    args = parser.parse_args()
    if args.mode == "full" and not args.confirm_full_training:
        parser.error("Full training requires a separate --confirm-full-training invocation; 4A never starts it")
    root = Path(args.output)
    if not root.resolve().is_relative_to((Path.cwd()/"artifacts/stage4").resolve()):
        raise DataError("INVALID_VALUE", "Stage4 output path required")
    started = perf_counter()
    with threadpool_limits(limits=2):
        engines = setup()
        catalogs,admission_seconds = load_catalogs(engines)
        load_seconds = perf_counter()-started
        experiments = ["O","C"] if args.experiment == "both" else [args.experiment]
        if args.mode == "smoke":
            if args.seed != 42:
                raise DataError("INVALID_VALUE", "4A smoke command is fixed seed42; multiseed full experiment remains deferred")
            smoke(catalogs,root,experiments)
        elif args.mode == "benchmark":
            benchmark(catalogs,root,load_seconds)
        elif args.mode == "estimate":
            refine_estimates(catalogs,root)
        else:
            for exp in experiments:
                cfg = RunConfig(kind="full",seed=args.seed,batch_size=256,max_epochs=100,variant=args.variant,method=args.method)
                train_model(catalogs[exp]["train"],catalogs[exp]["validation"],root/"full"/exp/f"{args.variant}_{args.method}_seed{args.seed}",cfg,root/"full/cache")
    print(json.dumps({"mode":args.mode,"elapsed_seconds":perf_counter()-started,"load_seconds":load_seconds,
                      "admission_seconds":admission_seconds,"memory":process_memory()}),flush=True)


if __name__ == "__main__":
    main()
