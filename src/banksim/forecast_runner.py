"""Resumable training. Only committed complete optimizer steps are recoverable."""

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from .cluster_io import fingerprint
from .data import DataError, sha256_file
from .data_stage import dump_json
from .forecast_cache import CompactReferences
from .forecast_checkpoint import load_checkpoint, model_lineage, publish_checkpoint
from .forecast_data import CustomerSampler
from .forecast_metrics import EarlyStopping
from .forecast_model import ConsumptionMLP, mask_reference, move_batch, network_inputs, optimizer_for, resolve_device, train_step
from .forecast_runtime import Progress, WriterLock
from .forecast_training import evaluate, prepare_batch
from .retrieval_io import read_json


def _epoch_stats():
    return {"loss_sum":0.,"sample_count":0,"masked":0,"prepare_seconds":0.,"train_seconds":0.}


def run_training(train, validation, output_dir, config, cache_root, *, resume=False,
                 checkpoint_every_batches=25, progress_seconds=30,
                 stop_after_steps=None, stop_after_epoch=None, validation_fault=None,
                 device="cpu"):
    device = resolve_device(device)
    use_cuda = device.type == "cuda"
    output_dir = Path(output_dir)
    if not output_dir.resolve().is_relative_to((Path.cwd()/"artifacts/stage4").resolve()):
        raise DataError("INVALID_VALUE", "Models must be saved inside artifacts/stage4")
    if checkpoint_every_batches < 1:
        raise DataError("INVALID_VALUE", "Checkpoint interval must be positive")
    if config.kind != "smoke" and (stop_after_steps or stop_after_epoch or validation_fault):
        raise DataError("INVALID_VALUE", "Test stops/faults are smoke-only")
    if train.partition != "train" or validation.partition != "validation" or train.experiment != validation.experiment:
        raise DataError("INVALID_VALUE", "Train/validation admission mismatch; test cannot select weights")
    train.validate(); validation.validate()
    expected_lengths = {0} if train.experiment == "O" else {1,3,7,14}
    if set(validation.frame.lc_key) != expected_lengths or validation.frame.empty:
        raise DataError("VALIDATION_EMPTY", "Required validation strata absent")
    with ExitStack() as stack:
        stack.enter_context(WriterLock(output_dir/"run.lock"))
        log = stack.enter_context(Progress(output_dir,train.experiment,config.variant,progress_seconds))
        log.update(device=str(device))
        if not resume and any((output_dir/n).exists() for n in ("config.json","best.pt","checkpoint_manifest.json")):
            raise DataError("INVALID_VALUE", "Run exists; use explicit resume or a new directory")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        sampler = CustomerSampler(train,config.seed)
        dropout_rng = torch.Generator().manual_seed(config.seed+1000)
        stopper = EarlyStopping(config.patience)
        epoch_size = sampler.epoch_size(config.batch_size)
        batches = epoch_size//config.batch_size
        curve,trace,stats = [],[],_epoch_stats()
        epoch,next_batch,steps,phase = 1,0,0,"training"
        previous_elapsed = 0.
        if resume:
            model,metadata,saved = load_checkpoint(output_dir,train.engine,kind=config.kind,
                                                   variant=config.variant,method=config.method,name="last",device=device)
            if (metadata["config"] != asdict(config) or metadata["seed"] != config.seed
                    or metadata.get("device", "cpu") != str(device)
                    or metadata["train_catalog"] != train.identity or metadata["validation_catalog"] != validation.identity
                    or read_json(output_dir/"config.json") != {k:v for k,v in metadata.items()
                                                               if k not in ("weight_sha256","checkpoint_format")}):
                raise DataError("CONFIG_MISMATCH", "Resume configuration or sample selection differs")
            if saved.get("resume_format") != 2 or saved.get("batches_per_epoch") != batches:
                raise DataError("CONFIG_MISMATCH", "Checkpoint does not contain compatible batch recovery state")
            optimizer = optimizer_for(model)
            optimizer.load_state_dict(saved["optimizer"])
            stopper.best,stopper.best_epoch,stopper.bad_epochs = saved["best_score"],saved["best_epoch"],saved["bad_epochs"]
            sampler.rng.bit_generator.state = saved["sampler_rng"]
            sampler.no_feasible_cold = saved["no_feasible_cold"]
            dropout_rng.set_state(saved["reference_dropout_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if use_cuda:
                if "cuda_rng" not in saved:
                    raise DataError("CHECKPOINT_INVALID", "CUDA recovery state is missing")
                torch.cuda.set_rng_state(saved["cuda_rng"],device=device)
            epoch,next_batch,steps,phase = saved["epoch"],saved["next_batch"],saved["completed_steps"],saved["phase"]
            curve,trace,stats = saved["curve"],saved["sampling_trace"],saved["epoch_stats"]
            previous_elapsed = saved["elapsed_seconds"]
            if (phase not in ("training","validating","epoch_complete") or not 0 <= next_batch <= batches
                    or steps != (epoch-1)*batches+next_batch or stats["sample_count"] != next_batch*config.batch_size
                    or len(curve) != epoch-(phase != "epoch_complete")):
                raise DataError("CHECKPOINT_INVALID", "Inconsistent recovery cursor/curve")
            log.emit("resumed",restored_epoch=epoch,next_batch=next_batch,completed_steps=steps,
                     best_epoch=stopper.best_epoch,bad_epochs=stopper.bad_epochs,restored_phase=phase)
        else:
            manifests = []
            for name,catalog in (("train",train),("validation",validation)):
                path = output_dir/f"{name}_selection.parquet"
                catalog.frame.to_parquet(path,index=False)
                manifests.append({"file":path.name,"sha256":sha256_file(path)})
            lineage = model_lineage(train.engine)
            metadata = {"kind":config.kind,"variant":config.variant,"method":config.method,"seed":config.seed,
                        "config":asdict(config),"lineage":lineage,"manifests":manifests,"device":str(device),
                        "train_catalog":train.identity,"validation_catalog":validation.identity,
                        "model_version":fingerprint({"run":asdict(config),"lineage":lineage,"device":str(device),
                                                     "train":train.identity,"validation":validation.identity})}
            dump_json(output_dir/"config.json",metadata)
            model = ConsumptionMLP().to(device)
            optimizer = optimizer_for(model)
        # Constructing caches does not draw random numbers.
        refs = stack.enter_context(CompactReferences(train,cache_root,config.method))
        val_refs = stack.enter_context(CompactReferences(validation,cache_root,config.method))
        log.counters = lambda: {"cache_hits":refs.hits+val_refs.hits,"cache_misses":refs.misses+val_refs.misses}
        started = perf_counter()
        last_generation = None

        def progress():
            log.update(epoch=epoch,batch=next_batch,batches=batches,phase=phase,
                       cache_hits=refs.hits+val_refs.hits,cache_misses=refs.misses+val_refs.misses,
                       last_checkpoint=last_generation)

        def commit(improved=False):
            nonlocal last_generation
            refs.flush(); val_refs.flush()
            payload = {"resume_format":2,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                       "epoch":epoch,"next_batch":next_batch,"completed_steps":steps,"phase":phase,
                       "batches_per_epoch":batches,"epoch_stats":stats,"best_epoch":stopper.best_epoch,
                       "best_score":stopper.best,"bad_epochs":stopper.bad_epochs,
                       "torch_rng":torch.get_rng_state(),"reference_dropout_rng":dropout_rng.get_state(),
                       "sampler_rng":sampler.rng.bit_generator.state,"no_feasible_cold":sampler.no_feasible_cold,
                       "curve":curve,"sampling_trace":trace,
                       "elapsed_seconds":previous_elapsed+perf_counter()-started}
            if use_cuda:
                payload["cuda_rng"] = torch.cuda.get_rng_state(device)
            last_generation = publish_checkpoint(output_dir,payload,metadata,improved=improved)
            progress()
            log.emit("checkpoint",completed_steps=steps)

        def pause():
            result = {"status":"SMOKE_STOPPED","epoch":epoch,"next_batch":next_batch,
                      "completed_steps":steps,"phase":phase,"best_epoch":stopper.best_epoch,
                      "bad_epochs":stopper.bad_epochs,"target_epochs":config.max_epochs}
            dump_json(output_dir/f"stop_step_{steps}.json",result)
            log.emit("planned_smoke_stop",**result)
            return result

        try:
            # A zero-step checkpoint also protects a first-batch interruption.
            if not resume:
                commit()
            else:
                from .forecast_checkpoint import committed_manifest
                last_generation = committed_manifest(output_dir)["last"]["generation"]
            while epoch <= config.max_epochs:
                if phase == "epoch_complete":
                    if stopper.bad_epochs >= config.patience or epoch == config.max_epochs:
                        break
                    epoch += 1
                    next_batch,stats,phase = 0,_epoch_stats(),"training"
                progress()
                while next_batch < batches:
                    tick = perf_counter()
                    keys = [sampler.draw() for _ in range(config.batch_size)]
                    batch,records,_ = prepare_batch(train,keys,refs,no_retrieval=config.variant == "no_retrieval")
                    batch,count = mask_reference(batch,training=True,no_retrieval=config.variant == "no_retrieval",generator=dropout_rng)
                    stats["prepare_seconds"] += perf_counter()-tick
                    tick = perf_counter()
                    loss = train_step(model,optimizer,move_batch(batch,device))  # Must finish before moving the cursor.
                    stats["loss_sum"] += loss*len(keys)
                    stats["sample_count"] += len(keys)
                    stats["masked"] += count
                    stats["train_seconds"] += perf_counter()-tick
                    if config.kind == "smoke":
                        trace.extend({"epoch":epoch,"batch":next_batch,"customer":k.customer,"origin_t":k.t,"Lc":k.Lc,
                                      "reference_before":r["reference_available"],"reference_after":int(batch["r_mask"][i,0])}
                                     for i,(k,r) in enumerate(zip(keys,records)))
                    next_batch += 1
                    steps += 1
                    progress()
                    if steps % checkpoint_every_batches == 0 or (stop_after_steps and steps >= stop_after_steps):
                        commit()
                    if stop_after_steps and steps >= stop_after_steps:
                        return pause()
                phase = "validating"
                commit()  # Validation can be rerun; curve/early stopping not yet updated.
                tick = perf_counter()
                validation_result = evaluate(model,validation,val_refs,variant=config.variant,batch_size=config.batch_size)
                score = validation_result["methods"]["model"]["selection_mse"]
                improved,_ = stopper.update(score,epoch)
                curve.append({"epoch":epoch,"scaled_training_mse":stats["loss_sum"]/stats["sample_count"],
                              "validation_mse":score,"best_epoch":stopper.best_epoch,"improved":improved,
                              "reference_masked":stats["masked"],"prepare_seconds":stats["prepare_seconds"],
                              "train_seconds":stats["train_seconds"],"validation_seconds":perf_counter()-tick})
                if validation_fault:
                    validation_fault(epoch)  # Test death after in-memory update but before commit.
                phase = "epoch_complete"
                commit(improved)
                dump_json(output_dir/"curve.json",curve)
                log.emit("epoch_complete",training_mse=curve[-1]["scaled_training_mse"],validation_mse=score,
                         best_epoch=stopper.best_epoch,bad_epochs=stopper.bad_epochs)
                if stop_after_epoch and epoch >= stop_after_epoch:
                    return pause()
            restored,_,_ = load_checkpoint(output_dir,train.engine,kind=config.kind,variant=config.variant,method=config.method,device=device)
            final_validation = evaluate(restored,validation,val_refs,variant=config.variant,batch_size=config.batch_size,collect=config.kind == "smoke")
            again,_,_ = load_checkpoint(output_dir,train.engine,kind=config.kind,variant=config.variant,method=config.method,device=device)
            probe,_,_ = prepare_batch(validation,[validation.key_at(0)],val_refs)
            probe,_ = mask_reference(probe,no_retrieval=config.variant == "no_retrieval")
            probe_inputs = move_batch(network_inputs(probe),device)
            with torch.no_grad():
                identical = torch.equal(restored(**probe_inputs),again(**probe_inputs))
            assert identical and final_validation["methods"]["model"]["selection_mse"] == stopper.best
            result = {"kind":config.kind,"experiment":train.experiment,"variant":config.variant,"seed":config.seed,"device":str(device),
                      "train_customers":len(train.customers),"ordinary_training_pairs":len(train.ordinary),
                      "saved_training_mode_rows":len(train.frame),"epoch_sample_count":epoch_size,"epochs_run":len(curve),
                      "completed_steps":steps,"best_epoch":stopper.best_epoch,"bad_epochs":stopper.bad_epochs,
                      "best_validation_mse":stopper.best,"best_restored":True,"reload_equal":identical,
                      "validation":final_validation,"curve":curve,"elapsed_seconds":previous_elapsed+perf_counter()-started,
                      "no_feasible_cold":sampler.no_feasible_cold,"cache_hits":refs.hits,"cache_misses":refs.misses,
                      "model_version":metadata["model_version"],"a":train.engine.bundle.amount_scale["a"]}
            dump_json(output_dir/"summary.json",result)
            if config.kind == "smoke":
                dump_json(output_dir/"sampling_trace.json",trace)
            return result
        except Exception as exc:
            log.emit("failed",error=str(exc))
            dump_json(output_dir/"FAILED.json",{"status":"FAILED","type":type(exc).__name__,"message":str(exc),"curve":curve})
            raise
