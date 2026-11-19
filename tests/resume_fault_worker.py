"""Independent-process crash/lock probes used by resume tests (not training data)."""
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from banksim.forecast_cache import CompactReferences
from banksim.forecast_checkpoint import publish_checkpoint
from banksim.forecast_data import SampleKey
from banksim.forecast_runtime import WriterLock


@dataclass
class Params:
    m: int = 5


class Catalog:
    identity = "storage-fault-fixture"
    experiment,partition = "O","train"
    frame = [0,1]
    def __init__(self):
        self.engine = SimpleNamespace(params=Params(),versions={"fixture":"1"},
            store=SimpleNamespace(amounts={"c":np.arange(80.)}),retrieve=self.retrieve)

    def locate(self,key):
        return key.t-59,SimpleNamespace(ell=1)

    def retrieve(self,*args,**kwargs):
        return {"R":[1.]*7,"r_log":np.log1p(np.ones(7)).tolist(),"reference_available":1,
                "candidate_count":1,"expanded_all":False,"library_version":"fixture",
                "candidates":[{"donor_id":"c","key_end":1,"distance":0.,"weight":1.}]}


def main():
    mode,path,stage = sys.argv[1:]
    if mode == "checkpoint":
        with WriterLock(Path(path)/"run.lock"):
            publish_checkpoint(path,{"step":2},{"fixture":True},improved=True,
                               fault=lambda s: os._exit(73) if s == stage else None)
    elif mode == "lock":
        with WriterLock(path):
            print("acquired",flush=True)
    elif mode == "cache":
        def fault(s):
            if s == stage:
                os._exit(74)
        cache = CompactReferences(Catalog(),path,fault=fault)
        cache.get(SampleKey("c",60))
        cache.close()
    elif mode == "cache-lock":
        with CompactReferences(Catalog(),path):
            print("acquired",flush=True)


if __name__ == "__main__":
    main()
