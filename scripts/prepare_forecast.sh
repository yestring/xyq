#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Python executable not found: %s. Create the project venv with python3.12 -m venv --copies .venv.\n' "$PYTHON" >&2
  exit 1
fi

if [[ -d artifacts/stage4 && -n "$(ls -A artifacts/stage4)" ]]; then
  printf 'Existing forecast runs depend on these artifacts. Use a fresh checkout to regenerate stages 1–3.\n' >&2
  exit 1
fi

mkdir -p artifacts/test-tmp
if command -v flock >/dev/null 2>&1; then
  exec 9>artifacts/.prepare.lock
  if ! flock -n 9; then
    printf 'Another preparation process is writing artifacts.\n' >&2
    exit 1
  fi
fi

"$PYTHON" - <<'PYTHON_PREFLIGHT'
import gzip
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

if not Path(sys.executable).resolve().is_relative_to(Path.cwd()):
    raise SystemExit(
        "The data stage records a project-relative Python executable. "
        "Create the project venv with python3.12 -m venv --copies .venv."
    )

raw = Path("data/raw/bs140513_032310.csv")
expected = "e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377"


def verify(path):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"BankSim SHA-256 mismatch: {path}")


if raw.exists():
    verify(raw)
else:
    archive = raw.with_suffix(".csv.gz")
    if not archive.is_file():
        raise SystemExit(f"Place the original BankSim CSV at {raw} or its gzip at {archive}.")
    with tempfile.NamedTemporaryFile(dir=raw.parent, delete=False) as output:
        tmp = Path(output.name)
        try:
            with gzip.open(archive, "rb") as compressed:
                shutil.copyfileobj(compressed, output)
            output.close()
            verify(tmp)
            tmp.replace(raw)
        finally:
            tmp.unlink(missing_ok=True)
print(f"BankSim input verified: {raw}")
PYTHON_PREFLIGHT

snapshot() {
  "$PYTHON" - "$@" <<'PYTHON_SNAPSHOT'
import hashlib
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
files = set()
for name in sys.argv[2:]:
    path = Path(name)
    if path.is_dir():
        files.update(child for child in path.rglob("*") if child.is_file())
    elif path.is_file():
        files.add(path)
    else:
        raise SystemExit(f"Missing protected input: {path}")

baseline = {}
for path in sorted(files):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    baseline[path.as_posix()] = {"sha256": digest, "bytes": path.stat().st_size}

destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PYTHON_SNAPSHOT
}

printf '\nStage 1: validate BankSim transactions and build the daily panel.\n'
"$PYTHON" -m banksim.data_stage --mode full
"$PYTHON" -m pytest -q tests/test_data.py tests/test_visibility.py \
  --basetemp artifacts/test-tmp/stage1 --acceptance-stage=1 \
  --acceptance-report artifacts/acceptance_data.json

protected=(data/raw/bs140513_032310.csv data/reference config docs examples prompts
           START_HERE.md manifest.json requirements-data.lock.txt
           artifacts/data artifacts/acceptance_data.json)
snapshot artifacts/stage2/baseline.json "${protected[@]}"

printf '\nStage 2: build O/C splits and the fitted clustering inputs.\n'
"$PYTHON" -m banksim.cluster_stage
"$PYTHON" -m pytest -q tests/test_stage2_unit.py tests/test_stage2_full.py \
  --basetemp artifacts/test-tmp/stage2 --acceptance-stage=2 \
  --acceptance-report artifacts/stage2/acceptance-cluster.json

snapshot artifacts/stage3/baseline.json "${protected[@]}" artifacts/stage2 artifacts/splits
cp -- DEVELOPMENT_PROGRESS.md artifacts/stage3/progress-before.txt

printf '\nStage 3: build reference retrieval samples and run acceptance.\n'
"$PYTHON" -m banksim.retrieval_stage --mode sample
"$PYTHON" -m pytest -q tests/test_retrieval.py tests/test_retrieval_full.py \
  --basetemp artifacts/stage3/test-tmp --acceptance-stage=3 \
  --acceptance-report artifacts/stage3/acceptance-final.json

"$PYTHON" - <<'PYTHON_ACCEPTANCE'
import json
from pathlib import Path

report = json.loads(Path("artifacts/stage3/acceptance-final.json").read_text(encoding="utf-8"))
statuses = report["acceptance"]
incomplete = [code for code, result in statuses.items() if result["status"] != "PASS"]
if not statuses or incomplete:
    raise SystemExit(f"Stage 3 acceptance is incomplete: {', '.join(incomplete)}")
PYTHON_ACCEPTANCE

printf '\nAccepted forecast inputs are ready under artifacts/.\n'
