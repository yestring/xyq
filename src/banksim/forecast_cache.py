"""Indexed fixed-width training cache, no per-query JSON or database service."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from .cluster_io import fingerprint
from .data import DataError
from .data_stage import dump_json
from .retrieval_io import read_json
from .forecast_runtime import WriterLock, io_path


ROW_DTYPE = np.dtype([("ready","u1"),("R","f8",7),("r_log","f8",7),
                      ("available","u1"),("count","u1"),("expanded","u1"),
                      ("donors","i4",5),("ends","i2",5),("distances","f8",5),
                      ("weights","f8",5),("library","S64"),("digest","S16")])


class CompactReferences:
    def __init__(self, catalog, root, method="same_group", *, fault=None):
        self.lock = None
        self.rows = None
        self.fault = fault
        if method not in ("same_group","all") or catalog.engine.params.m != 5:
            raise DataError("CONFIG_MISMATCH", "Compact v1 cache uses exact production m=5")
        root = Path(root)
        if not root.resolve().is_relative_to((Path.cwd()/"artifacts/stage4").resolve()):
            raise DataError("INVALID_VALUE", "Training cache must be inside artifacts/stage4")
        self.catalog, self.method = catalog, method
        self.identity = {"format": "compact-references-v1", "catalog": catalog.identity,
                         "versions": catalog.engine.versions, "parameters": asdict(catalog.engine.params),
                         "experiment": catalog.experiment, "partition": catalog.partition, "method": method,
                         "library_mode": "frozen"}
        self.root = io_path(root / fingerprint(self.identity))
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = WriterLock(self.root/"writer.lock", "CACHE_LOCKED")
        self.donors = sorted(catalog.engine.store.amounts)
        self.donor_index = {c:i for i,c in enumerate(self.donors)}
        metadata = {"identity":self.identity,"rows":len(catalog.frame),"dtype":str(ROW_DTYPE),"donors":self.donors}
        path = self.root/"references.npy"
        if (self.root/"metadata.json").exists():
            if read_json(self.root/"metadata.json") != metadata:
                raise DataError("CONFIG_MISMATCH", "Compact cache lineage differs")
            self.rows = np.load(path,mmap_mode="r+",allow_pickle=False)
            if self.rows.dtype != ROW_DTYPE or self.rows.shape != (len(catalog.frame),):
                raise DataError("CONFIG_MISMATCH", "Compact cache shape differs")
        else:
            self.rows = np.lib.format.open_memmap(path,mode="w+",dtype=ROW_DTYPE,shape=(len(catalog.frame),))
            self.rows[:] = np.zeros((),dtype=ROW_DTYPE)
            self.rows.flush()
            dump_json(self.root/"metadata.json",metadata)
        self.hits = self.misses = 0

    @staticmethod
    def digest(row):
        return hashlib.blake2b(row.tobytes()[1:-16],digest_size=16).digest()

    def get(self, key):
        i, admitted = self.catalog.locate(key)
        row = self.rows[i]
        if int(row["ready"]) not in (0,1):
            raise DataError("CONFIG_MISMATCH", "Compact cache invalid commit marker")
        hit = bool(row["ready"])
        if row["ready"]:
            if row["digest"].tobytes().ljust(16,b"\0") != self.digest(row):
                raise DataError("CONFIG_MISMATCH", "Compact cache row checksum differs")
            self.hits += 1
        else:
            e = self.catalog.engine
            full = e.retrieve(key.customer,key.t,Lc=key.Lc,partition=self.catalog.partition,method=self.method)
            row = np.zeros((),dtype=ROW_DTYPE)
            row["R"], row["r_log"] = full["R"],full["r_log"]
            row["available"], row["count"], row["expanded"] = full["reference_available"],full["candidate_count"],full["expanded_all"]
            row["donors"] = -1
            for j,c in enumerate(full["candidates"]):
                row["donors"][j],row["ends"][j] = self.donor_index[c["donor_id"]],c["key_end"]
                row["distances"][j],row["weights"][j] = c["distance"],c["weight"]
            row["library"] = full["library_version"].encode()
            row["digest"] = self.digest(row)
            # Publish payload with ready=0 first. An interrupted row is recomputed;
            # ready=1 is accepted only with the complete payload checksum.
            self.rows[i] = row
            if self.fault:
                self.rows.flush()
                self.fault("row_payload_written")
            self.rows[i]["ready"] = 1
            if self.fault:
                self.rows.flush()
                self.fault("row_committed")
            row = self.rows[i]
            self.misses += 1
        ell = int(admitted.ell)
        Q = self.catalog.engine.store.amounts[key.customer][key.t-ell+1:key.t+1].tolist()
        return {"Q":Q,"R":row["R"].tolist(),"r_log":row["r_log"].tolist(),
                "reference_available":int(row["available"]),"candidate_count":int(row["count"]),
                "expanded_all":bool(row["expanded"]),"cache_hit":hit,
                "trace":self.trace(row,ell)}

    def trace(self,row,ell):
        records = []
        library = row["library"].decode()
        for j in range(int(row["count"])):
            donor,end = self.donors[int(row["donors"][j])],int(row["ends"][j])
            records.append({"candidate_id":f"{self.catalog.experiment}:{library}:{donor}:{end-ell+1}:{end}:{end+7}",
                            "donor_id":donor,"key_start":end-ell+1,"key_end":end,"value_start":end+1,"value_end":end+7,
                            "distance":float(row["distances"][j]),"weight":float(row["weights"][j])})
        return records

    def flush(self):
        self.rows.flush()

    def close(self):
        if self.rows is not None:
            self.rows.flush()
            self.rows._mmap.close()
            self.rows = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
