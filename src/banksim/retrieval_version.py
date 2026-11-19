"""Explicit retrieval dependencies; unrelated forecaster modules do not invalidate R."""

import json
import importlib.metadata
from pathlib import Path

from .cluster_io import fingerprint
from .data import sha256_file


def semantic_version(root=Path("src/banksim"), compatibility=Path("config/retrieval_compatibility.json")):
    saved = json.loads(Path(compatibility).read_text(encoding="utf-8"))
    actual = {name: sha256_file(Path(root) / name) for name in saved["dependency_hashes"]}
    packages = {name:importlib.metadata.version(name) for name in saved["packages"]}
    if actual == saved["dependency_hashes"] and packages == saved["packages"]:
        # This compatibility baseline is tested against the saved stage-three traces.
        return saved["stage3_implementation_version"]
    return fingerprint({"policy": "retrieval-dependencies-v1", "files": actual,"packages":packages})
