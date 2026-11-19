"""The single specified float32 dual-branch MLP and input conversion."""

import os

import numpy as np
import torch
from torch import nn

from .data import DataError


def resolve_device(value):
    """Validate the requested runtime, with no CPU fallback for CUDA."""
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DataError("INVALID_VALUE", "Device must be cpu, cuda or cuda:N") from exc
    if device == torch.device("cpu"):
        return device
    if device.type != "cuda":
        raise DataError("INVALID_VALUE", "Device must be cpu, cuda or cuda:N")
    # Required by deterministic CUDA linear layers; set it before CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise DataError("CUDA_UNAVAILABLE", "The requested CUDA device is not available")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise DataError("CUDA_UNAVAILABLE", "The CUDA device index is not visible", index=index)
    return torch.device("cuda", index)


def move_batch(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


class ConsumptionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = nn.Sequential(nn.Linear(56,64,bias=True), nn.ReLU())
        self.reference = nn.Sequential(nn.Linear(8,32,bias=True), nn.ReLU())
        self.joint = nn.Sequential(nn.Linear(96,64,bias=True), nn.ReLU(), nn.Linear(64,7,bias=True), nn.Softplus())

    def forward(self, q_log, q_mask, r_log, r_mask):
        batch = q_log.shape[0]
        for x, width in ((q_log,28),(q_mask,28),(r_log,7),(r_mask,1)):
            if x.dtype != torch.float32 or x.shape != (batch,width) or not torch.isfinite(x).all():
                raise DataError("INVALID_VALUE", "Predictor expects finite float32 tensors with fixed shapes")
        h = self.history(torch.cat((q_log,q_mask),dim=1))
        r = self.reference(torch.cat((r_log,r_mask),dim=1))
        return self.joint(torch.cat((h,r),dim=1))


def tensor_inputs(records, targets_cents=None, a=1.0):
    if not np.isfinite(a) or a <= 0:
        raise DataError("CONFIG_MISMATCH", "Amount scale must be positive")
    n = len(records)
    q_log, q_mask = np.zeros((n,28),np.float32), np.zeros((n,28),np.float32)
    r_log, r_mask = np.zeros((n,7),np.float32), np.zeros((n,1),np.float32)
    for i,r in enumerate(records):
        q = np.asarray(r["Q"],dtype=float)
        if q.ndim != 1 or not 1 <= len(q) <= 28 or not np.isfinite(q).all() or (q < 0).any():
            raise DataError("INVALID_VALUE", "No valid observed query")
        q_log[i,-len(q):] = np.log1p(q)
        q_mask[i,-len(q):] = 1
        if r["reference_available"]:
            r_log[i] = r["r_log"]  # Already log1p(R); do not log again.
            r_mask[i] = 1
    result = {k:torch.from_numpy(v) for k,v in (("q_log",q_log),("q_mask",q_mask),("r_log",r_log),("r_mask",r_mask))}
    if targets_cents is not None:
        y = np.asarray(targets_cents,dtype=float)/100/a
        if y.shape != (n,7) or not np.isfinite(y).all() or (y < 0).any():
            raise DataError("INVALID_VALUE", "Invalid target cents")
        result["y_scaled"] = torch.tensor(y,dtype=torch.float32)
    return result


def mask_reference(batch, *, training=False, no_retrieval=False, generator=None):
    result = {k:v.clone() for k,v in batch.items()}
    present = result["r_mask"][:,0].bool()
    hidden = present if no_retrieval else (present & (torch.rand(len(present),generator=generator) < .2)) if training else torch.zeros_like(present)
    result["r_log"][hidden] = 0
    result["r_mask"][hidden] = 0
    return result, int(hidden.sum())


def network_inputs(batch):
    return {k:batch[k] for k in ("q_log","q_mask","r_log","r_mask")}


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**network_inputs(batch))
    loss = torch.mean((output-batch["y_scaled"])**2)
    if not torch.isfinite(loss):
        raise DataError("TRAINING_FAILED", "Nonfinite training loss")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise DataError("TRAINING_FAILED", "Nonfinite or missing gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise DataError("TRAINING_FAILED", "Nonfinite optimizer parameters")
    return float(loss.detach())
