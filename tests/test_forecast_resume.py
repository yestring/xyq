import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from banksim.data import DataError
from banksim.forecast_cache import CompactReferences
from banksim.forecast_checkpoint import checkpoint_files, committed_manifest, publish_checkpoint
from banksim.forecast_data import SampleKey
from banksim.forecast_runtime import Progress, WriterLock
from resume_fault_worker import Catalog


def child(mode,path,stage="none"):
    return subprocess.run([sys.executable,"tests/resume_fault_worker.py",mode,str(path),stage],
                          capture_output=True,text=True,timeout=90)


@pytest.mark.parametrize("stage",["weights_written","metadata_written","before_commit","after_commit"])
def test_checkpoint_process_death_keeps_complete_commit(tmp_path,stage):
    with WriterLock(tmp_path/"run.lock"):
        publish_checkpoint(tmp_path,{"step":1},{"fixture":True},improved=True)
    result = child("checkpoint",tmp_path,stage)
    assert result.returncode == 73,result.stderr
    weights,_ = checkpoint_files(tmp_path,"last")
    assert torch.load(weights,weights_only=True)["step"] == (2 if stage == "after_commit" else 1)
    assert committed_manifest(tmp_path)["best"] == committed_manifest(tmp_path)["last"]
    # The process died while holding the run lock; a fresh process can acquire it.
    assert child("lock",tmp_path/"run.lock").returncode == 0


def test_corrupt_latest_rolls_back_but_uncommitted_files_never_count(tmp_path):
    orphan = tmp_path/"checkpoints"
    orphan.mkdir()
    torch.save({"step":999},orphan/"uncommitted.pt")
    (orphan/"uncommitted.json").write_text("{}")
    with pytest.raises(DataError) as exc:
        checkpoint_files(tmp_path)
    assert exc.value.code == "MODEL_NOT_TRAINED"
    with WriterLock(tmp_path/"run.lock"):
        publish_checkpoint(tmp_path,{"step":1},{},improved=True)
        publish_checkpoint(tmp_path,{"step":2},{},improved=True)
    weights,_ = checkpoint_files(tmp_path,"last")
    weights.write_bytes(b"incomplete")
    valid,_ = checkpoint_files(tmp_path,"last")
    assert torch.load(valid,weights_only=True)["step"] == 1


@pytest.mark.parametrize("stage",["row_payload_written","row_committed"])
def test_cache_process_death_reuses_complete_and_recomputes_incomplete(tmp_path,stage):
    with CompactReferences(Catalog(),tmp_path) as cache:
        cache.get(SampleKey("c",59))
    result = child("cache",tmp_path,stage)
    assert result.returncode == 74,result.stderr
    with CompactReferences(Catalog(),tmp_path) as cache:
        assert cache.get(SampleKey("c",59))["cache_hit"]
        row = cache.get(SampleKey("c",60))
        assert row["cache_hit"] == (stage == "row_committed")
        assert row["R"] == [1.]*7
        assert row["trace"][0]["weight"] == 1.


def test_cache_bad_committed_row_is_explicitly_rejected(tmp_path):
    with CompactReferences(Catalog(),tmp_path) as cache:
        cache.get(SampleKey("c",59))
        cache.rows[0]["R"][0] = 99.
    with CompactReferences(Catalog(),tmp_path) as cache:
        with pytest.raises(DataError,match="checksum"):
            cache.get(SampleKey("c",59))


def test_two_process_writers_are_rejected_and_logged(tmp_path):
    with WriterLock(tmp_path/"run.lock"):
        result = child("lock",tmp_path/"run.lock")
        assert result.returncode != 0 and '"code": "RUN_LOCKED"' in result.stdout
    with CompactReferences(Catalog(),tmp_path/"cache"):
        result = child("cache-lock",tmp_path/"cache")
        assert result.returncode != 0 and '"code": "CACHE_LOCKED"' in result.stdout
    (tmp_path/"conflict.log").write_text(result.stdout+result.stderr,encoding="utf-8")


def test_heartbeat_runs_during_a_long_batch(tmp_path):
    with Progress(tmp_path,"O","retrieval",interval=.05) as progress:
        progress.update(epoch=1,batch=2,batches=3,last_checkpoint="committed",cache_hits=7,cache_misses=8)
        time.sleep(.23)
    records = [json.loads(s) for s in (tmp_path/"progress.jsonl").read_text().splitlines()]
    beats = [r for r in records if r["event"] == "progress"]
    assert len(beats) >= 2
    assert all(r["batch"] == 2 and r["last_checkpoint"] == "committed" for r in beats)
    assert max(b["elapsed_seconds"]-a["elapsed_seconds"] for a,b in zip(records,records[1:])) < 60
