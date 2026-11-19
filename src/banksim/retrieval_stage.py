"""Reproducible small-batch retrieval evidence, explicitly not a full experiment."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
from time import perf_counter

import pandas as pd

from .clustering import assign_profile
from .data import sha256_file
from .data_stage import dump_json, json_default
from .retrieval import RetrievalEngine
from .retrieval_io import load_engines, read_json, toy_engine


BASE = Path("artifacts/stage3")


def sample_requests(engines):
    requests = []
    # Deterministic systematic coverage, not random or representative sampling.
    for experiment in ("O", "C"):
        for partition in ("train", "validation", "test"):
            table = pd.read_parquet(f"artifacts/stage2/full/{experiment}/samples_{partition}.parquet")
            table["lc_group"] = table.Lc.fillna(0).astype(int)
            for lc, rows in table.groupby("lc_group", sort=True):
                chosen = rows.sort_values(["customer", "origin_t"]).drop_duplicates("customer").head(2)
                for row in chosen.itertuples():
                    methods = ("same_group", "all") if row.Tobs <= 14 else ("same_group",)
                    for method in methods:
                        requests.append(dict(cohort="main_sample", experiment=experiment, partition=partition,
                                             customer=row.customer, t=int(row.origin_t), Lc=None if lc == 0 else int(lc), method=method))
        engine = engines[experiment]
        customer = next(c for c in sorted(engine.donor_groups) if engine.store.first[c] == 0)
        for t in (61, 62, 89):
            requests.append(dict(cohort="boundary_diagnostic", experiment=experiment, partition="train",
                                 customer=customer, t=t, Lc=None, method="same_group"))
        customer = next(c for c in sorted(engine.source.panels) if engine.store.first[c] > 0)
        requests.append(dict(cohort="boundary_diagnostic", experiment=experiment, partition="train", customer=customer,
                             t=engine.store.first[customer]-1, Lc=None, method="same_group"))
        # Frozen production groups are too large to force expansion under m=5 here.
        # Use a separately labelled sensitivity run; never change production defaults or mix its rates.
        table = pd.read_parquet(f"artifacts/stage2/full/{experiment}/samples_train.parquet")
        candidates = table.loc[table.Lc.eq(14)].sort_values(["customer", "origin_t"]).drop_duplicates("customer")
        for row in candidates.itertuples():
            profile = engine.source.describe(row.customer, int(row.origin_t), Lc=14)
            group = assign_profile(engine.bundle, profile)
            if group["group_status"] == "FULL" and group["business_code"] == "B":
                requests.append(dict(cohort="expansion_sensitivity_m1500", experiment=experiment, partition="train",
                                     customer=row.customer, t=int(row.origin_t), Lc=14, method="same_group"))
                break
        else:
            raise RuntimeError("No actual FULL growth-group query found for expansion diagnostic")
    for i, request in enumerate(requests):
        request["request_id"] = f"q{i:03}"
    return requests


def metrics(results):
    frame = pd.DataFrame([{**{k: r[k] for k in ("experiment", "cohort", "method", "customer", "origin_t", "Lc")},
                           "has_history": r["Tobs"] > 0, "available": r["reference_available"],
                           "expanded": r["expanded_all"], "no_history": r["Tobs"] == 0} for r in results])
    summaries = []
    for keys, part in frame.groupby(["cohort", "experiment"], dropna=False, sort=True):
        n, eligible = len(part), int(part.has_history.sum())
        summaries.append(dict(cohort=keys[0], experiment=keys[1], calls=n,
                              unique_queries=len(part.drop_duplicates(["customer", "origin_t", "Lc"])),
                              customers=part.customer.nunique(), history_calls=eligible,
                              no_history_count=int(part.no_history.sum()), available_count=int(part.available.sum()),
                              expanded_count=int(part.expanded.sum()), no_reference_count=int((1-part.available).sum()),
                              effective_retrieval_rate=float(part.available.mean()), expansion_rate=float(part.expanded.mean()),
                              no_reference_rate=float((1-part.available).mean()),
                              history_only_retrieval_rate=float(part.available.sum()/eligible) if eligible else None))
    return summaries


def run(mode):
    started = perf_counter()
    BASE.mkdir(parents=True, exist_ok=True)
    toy = toy_engine().retrieve("T", 3)
    assert toy["R"] == [20., 30.] and [c["weight"] for c in toy["candidates"]] == [.5, .5]
    dump_json(BASE / "toy_result.json", toy)
    print("toy: distances=0,0; weights=0.5,0.5; R=[20,30]", flush=True)
    if mode == "toy":
        return
    engines = load_engines()
    print("loaded accepted O/C artifacts; no cluster fit invoked", flush=True)
    requests = sample_requests(engines)
    dump_json(BASE / "requests.json", {"sampling": "Each experiment/partition/Lc mode: lexicographically first two distinct customers, earliest legal origin; both methods for CROSS. Additional boundary and m1500 diagnostics are separate cohorts.",
                                       "requests": requests})
    results = []
    sensitivity = {}
    for request in requests:
        e = engines[request["experiment"]]
        if request["cohort"] == "expansion_sensitivity_m1500":
            e = sensitivity.setdefault(e.experiment, RetrievalEngine(e.experiment, e.source, e.bundle, e.donor_groups,
                    read_json(f"artifacts/splits/{e.experiment}.json")["ids"]["train"], e.versions,
                    params=replace(e.params, m=1500), store=e.store, cache=e.cache))
        start = perf_counter()
        kwargs = {k: request[k] for k in ("Lc", "partition", "method")}
        r = e.retrieve(request["customer"], request["t"], **kwargs)
        elapsed = perf_counter()-start
        replay = e.retrieve(request["customer"], request["t"], **kwargs)
        assert replay["cache_hit"]
        assert {k:v for k,v in r.items() if k != "cache_hit"} == {k:v for k,v in replay.items() if k != "cache_hit"}
        r.update(request_id=request["request_id"], cohort=request["cohort"], elapsed_seconds=elapsed,
                 replay_equal=True, replay_cache_hit=replay["cache_hit"])
        results.append(r)
        print(f"{request['request_id']} {e.experiment} {r['retrieval_path']} {r['group_status']} {r['selection_status']} n={r['candidate_count']} {elapsed:.3f}s", flush=True)
    with (BASE / "query_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=json_default)+"\n")
    baseline = read_json(BASE / "baseline.json")
    integrity = {path: sha256_file(path) == saved["sha256"] for path, saved in baseline.items()}
    if not all(integrity.values()):
        raise RuntimeError("Protected upstream files changed")
    report = {"kind": "small_reproducible_query_batch_not_full_experiment", "stage": 3,
              "elapsed_seconds": perf_counter()-started, "python": platform.python_version(),
              "metrics_denominator": "Every actual call, including method controls; unique query counts reported separately. NO_HISTORY included in all-call no-reference rate; history-only retrieval rate is also reported.",
              "metrics": metrics(results), "total_calls": len(results), "cache_replay_equal": all(r["replay_equal"] for r in results),
              "first_pass_cache_hits": sum(r["cache_hit"] for r in results),
              "max_distance_block_rows": max(r["max_distance_block_rows"] for r in results),
              "max_candidates_scored_for_one_query": max(r["scored_candidate_count"] for r in results),
              "upstream_integrity": integrity, "versions": {k:e.versions for k,e in engines.items()},
              "warnings": ["Small deterministic coverage sample; rates do not estimate full-data performance.",
                           "m=1500 expansion sensitivity is separate from default m=5 main results.",
                           "Growth-group name follows age order; observed low spending/activity is retained and does not validate a wealth-stage interpretation.",
                           "Retrieval weights are not confidence probabilities. R is a reference, not a model prediction."]}
    dump_json(BASE / "sample_report.json", report)
    print(json.dumps({"total_calls": len(results), "elapsed_seconds": report["elapsed_seconds"],
                      "upstream_files_unchanged": len(integrity), "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("toy", "sample"), default="sample")
    run(parser.parse_args().mode)
