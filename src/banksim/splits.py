"""Experiment IDs and legal history/label windows; no retrieval or training."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import DataError

SAMPLE_COLUMNS = ["experiment", "partition", "customer", "origin_t", "history_mode", "Lc",
                  "history_start", "history_end", "attribute_start", "attribute_end", "Tobs",
                  "ell", "H", "target_start", "target_end"]


def strict_id_split(customer_ids, seed=42):
    ids = np.array(sorted(set(customer_ids)), dtype=object)
    if any(not isinstance(value, str) or not value for value in ids):
        raise DataError("INVALID_VALUE", "Customer IDs must be nonempty strings")
    np.random.default_rng(seed).shuffle(ids)
    ntrain, nval = math.floor(0.7 * len(ids)), math.floor(0.15 * len(ids))
    return {"train": ids[:ntrain].tolist(), "validation": ids[ntrain:ntrain + nval].tolist(),
            "test": ids[ntrain + nval:].tolist()}


class SampleIndex:
    """Use prefix counts of unobserved/missing days to check every whole window."""

    def __init__(self, panel, registry, coverage_end=179):
        if panel.duplicated(["customer", "step"]).any():
            raise DataError("DATA_GAP", "Duplicate customer/day in sample input")
        self.coverage_end = coverage_end
        self.first = registry.set_index("customer")["first_seen_step"].astype(int).to_dict()
        self.invalid = {}
        for customer, rows in panel.groupby("customer", sort=False):
            rows = rows.set_index("step").reindex(range(coverage_end + 1))
            valid = rows["observed"].eq(True) & rows["target_cents"].notna()
            self.invalid[customer] = np.r_[0, (~valid).to_numpy(dtype=np.int64).cumsum()]

    def legal(self, customer, start, end):
        if customer not in self.invalid or start < 0 or end > self.coverage_end or start > end:
            return False
        prefix = self.invalid[customer]
        return bool(prefix[end + 1] == prefix[start])

    def ordinary_train_ids(self, cfg):
        return sorted(customer for customer, first in self.first.items()
                      if first <= cfg["split"]["train_label_end"] and any(
                          self.legal(customer, first, t) and self.legal(customer, t + 1, t + cfg["forecast"]["H"])
                          for t in origin_range(cfg, "train")))

    def samples(self, experiment, partition, ids, cfg):
        horizon, length = cfg["forecast"]["H"], cfg["forecast"]["L"]
        modes = [None, *cfg["forecast"]["cold_lengths"]] if partition == "train" else (
            [None] if experiment == "O" else cfg["forecast"]["cold_lengths"])
        output = []
        for customer in sorted(ids):
            first = self.first[customer]
            for t in origin_range(cfg, partition):
                if first > t or not self.legal(customer, t + 1, t + horizon):
                    continue
                # Later training samples are drawn from legal ordinary (customer,t) pairs.
                if partition == "train" and not self.legal(customer, first, t):
                    continue
                for lc in modes:
                    start = first if lc is None else t - lc + 1
                    if start < first or not self.legal(customer, start, t):
                        continue
                    tobs = t - start + 1
                    output.append((experiment, partition, customer, t, "ordinary" if lc is None else "cold",
                                   lc, start, t, start, t, tobs, min(length, tobs), horizon, t + 1, t + horizon))
        result = pd.DataFrame(output, columns=SAMPLE_COLUMNS)
        result["Lc"] = result["Lc"].astype("Int64")
        for field in ("experiment", "partition", "customer", "history_mode"):
            result[field] = result[field].astype("string")
        for field in set(SAMPLE_COLUMNS) - {"experiment", "partition", "customer", "history_mode", "Lc"}:
            result[field] = result[field].astype("int64")
        return result


def origin_range(cfg, partition):
    split = cfg["split"]
    return range(split[f"{partition}_origin_min"], split[f"{partition}_origin_max"] + 1)


def build_id_split(index, experiment, cfg):
    if experiment == "C":
        split = strict_id_split(index.first, cfg["split"]["split_seed"])
        sets = [set(split[part]) for part in ("train", "validation", "test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise DataError("INVALID_VALUE", "Strict ID groups overlap")
        return {"experiment": "C", "ids": split, "excluded_ids": [], "strict_disjoint": True}
    if experiment != "O":
        raise DataError("INVALID_VALUE", "Experiment must be O or C")
    train = index.ordinary_train_ids(cfg)
    excluded = sorted(set(index.first) - set(train))
    return {"experiment": "O", "ids": {part: train.copy() for part in ("train", "validation", "test")},
            "excluded_ids": excluded, "excluded_reason": "No legal ordinary training window",
            "strict_disjoint": False}


def sample_summary(frame, assigned_ids):
    by_mode = []
    for (mode, lc), rows in frame.groupby(["history_mode", "Lc"], dropna=False, sort=True):
        by_mode.append({"history_mode": mode, "Lc": None if pd.isna(lc) else int(lc),
                        "samples": len(rows), "customers": rows.customer.nunique(),
                        "origin_min": int(rows.origin_t.min()), "origin_max": int(rows.origin_t.max()),
                        "target_min": int(rows.target_start.min()), "target_max": int(rows.target_end.max())})
    return {"samples": len(frame), "customers": frame.customer.nunique(), "by_mode": by_mode,
            "assigned_without_samples": sorted(set(assigned_ids) - set(frame.customer))}
