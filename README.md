# BankSim 冷启动现金流预测

系统从公开的 BankSim 模拟交易构建客户日面板，按时间可见性生成样本，再按客群从历史中检索消费参考。预测模型是双分支 MLP，结合近期消费和检索参考预测未来 7 天的逐日消费。实验 O 使用常规历史，实验 C 按 1、3、7、14 天可用历史评估冷启动。

## Linux 单卡训练

从仓库根目录启动实验 C。脚本会安装依赖、验证 CUDA 前向和反向传播、准备数据并启动完整训练。

```bash
GPU=0 bash scripts/run_l20.sh
```

如果开发机需要代理下载依赖，在运行时设置 `PIP_DOWNLOAD_PROXY`。训练日志写入 `artifacts/stage4/full/C/retrieval_same_group_seed42_l20/train.log`。下面列出等价的手动流程，便于逐步排查。

以下以 Python 3.12、NVIDIA L20 和 CUDA 12.9 版 PyTorch 2.11.0 为例。使用 `--copies`，让数据验收能记录仓库内的解释器路径。数据已压缩随仓库提供，准备脚本会校验其 SHA-256。

```bash
python3.12 -m venv --copies .venv
.venv/bin/python -m pip install -r requirements-cluster.lock.txt
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu129 'torch==2.11.0'

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

.venv/bin/python - <<'PY'
import torch

torch.use_deterministic_algorithms(True)
x = torch.ones((8, 8), device="cuda:0", requires_grad=True)
(x @ x).sum().backward()
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0), torch.__version__)
PY

bash scripts/prepare_forecast.sh

RUN=artifacts/stage4/full/C/retrieval_same_group_seed42_l20

.venv/bin/python -m banksim.forecast_train_cli new \
  --experiment C --kind full --seed 42 \
  --variant retrieval --method same_group --device cuda:0 \
  --run-dir "$RUN" --checkpoint-every-batches 25 --progress-seconds 30 \
  --confirm-full-training
```

`CUDA_VISIBLE_DEVICES=0` 选择物理卡，可改为开发机上分配给你的卡号或 UUID。只暴露一张卡时，训练进程内仍使用 `--device cuda:0`。当显式选用 CUDA 且设备不可用时，程序会报错。

`prepare_forecast.sh` 在 CPU 上生成并验收数据、聚类和检索的产物。同一份验收产物可以供后续训练复用。训练的模型和 batch 使用 L20，样本构建、检索和磁盘缓存仍由 CPU 处理。历史进展和交接规格见 `DEVELOPMENT_PROGRESS.md` 和 `docs/`。

## 进度和恢复

训练目录的 `progress.jsonl` 记录设备、轮次和 checkpoint 进度，`summary.json` 保存最终验证指标。中断后保留准备产物和同一个训练目录，用相同参数把 `new` 改为 `resume`。本版本首次运行请新建训练目录，checkpoint 会核对源码版本和设备。

数据为 Edgar Lopez-Rojas 发布的 [BankSim v1](https://www.kaggle.com/datasets/ealaxi/banksim1)，许可为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。压缩包解压后与原始 CSV 字节一致，完整来源见 [数据说明](docs/DATA_SOURCE.md)。
