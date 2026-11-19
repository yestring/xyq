"""Content-addressed JSON query results and compact NumPy date indexes."""

import json
from pathlib import Path

import numpy as np

from .candidates import CandidateBank
from .cluster_io import fingerprint
from .data import DataError, sha256_file
from .data_stage import dump_json


class RetrievalCache:
    def __init__(self, root):
        self.root = Path(root)
        if not any(self.root.resolve().is_relative_to((Path.cwd() / folder).resolve())
                   for folder in ("artifacts/stage3", "artifacts/stage4")):
            raise DataError("INVALID_VALUE", "Retrieval caches must be inside artifacts/stage3 or stage4")
        for name in ("queries", "libraries"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def get(self, key):
        path = self.root / "queries" / (fingerprint(key) + ".json")
        if not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["key"] != key or fingerprint(saved["result"]) != saved["result_hash"]:
            raise DataError("CONFIG_MISMATCH", "Corrupt retrieval cache", path=path.as_posix())
        return saved["result"]

    def put(self, key, result):
        dump_json(self.root / "queries" / (fingerprint(key) + ".json"),
                  {"key": key, "result": result, "result_hash": fingerprint(result)})

    def bank(self, key, factory):
        base = self.root / "libraries" / fingerprint(key)
        meta_path, array_path = base.with_suffix(".json"), base.with_suffix(".npz")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["key"] != key or sha256_file(array_path) != meta["arrays_sha256"]:
                raise DataError("CONFIG_MISMATCH", "Corrupt candidate index")
            with np.load(array_path, allow_pickle=False) as arrays:
                return CandidateBank(tuple(meta["donors"]), arrays["donor_index"], arrays["key_end"],
                                     key["ell"], key["H"], key["cutoff"], fingerprint(key), meta["groups"])
        bank = factory()
        temp = base.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, donor_index=bank.donor_index, key_end=bank.key_end)
        temp.replace(array_path)
        dump_json(meta_path, {"key": key, "donors": bank.donors, "groups": bank.groups,
                              "arrays_sha256": sha256_file(array_path), "candidate_count": len(bank)})
        return bank
