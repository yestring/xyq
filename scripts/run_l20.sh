#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
GPU="${GPU:-0}"
PIP_DOWNLOAD_PROXY="${PIP_DOWNLOAD_PROXY:-}"
RUN=artifacts/stage4/full/C/retrieval_same_group_seed42_l20

if [[ -f "$RUN/config.json" ]]; then
  printf 'Training directory already exists: %s. Use the resume command in README.md.\n' "$RUN" >&2
  exit 1
fi

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
unset PYTHONPATH
python3.12 -m venv --copies .venv
PY="$PWD/.venv/bin/python"

pip_install() {
  "$PY" -m pip install --disable-pip-version-check --timeout 30 --retries 0 "$@"
}

pip_with_proxy() {
  if [[ -z "$PIP_DOWNLOAD_PROXY" ]]; then
    return 1
  fi
  pip_install --proxy "$PIP_DOWNLOAD_PROXY" "$@"
}

pip_install -r requirements-cluster.lock.txt ||
  pip_with_proxy --index-url https://pypi.org/simple -r requirements-cluster.lock.txt

TORCH_INDEX=https://download.pytorch.org/whl/cu129
pip_install --index-url "$TORCH_INDEX" 'torch==2.11.0+cu129' ||
  pip_with_proxy --index-url "$TORCH_INDEX" 'torch==2.11.0+cu129'
"$PY" -m pip check

export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src"

"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check GPU allocation and driver libraries.")
torch.use_deterministic_algorithms(True)
x = torch.ones((32, 32), device="cuda:0", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
print("CUDA OK:", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY

PYTHON="$PY" bash scripts/prepare_forecast.sh

mkdir -p "$RUN"
"$PY" -m banksim.forecast_train_cli new \
  --experiment C --kind full --seed 42 \
  --variant retrieval --method same_group --device cuda:0 \
  --run-dir "$RUN" --checkpoint-every-batches 25 --progress-seconds 30 \
  --confirm-full-training 2>&1 | tee "$RUN/train.log"
