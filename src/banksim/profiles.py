"""As-of profiles, frozen warmup transforms and the independent amount scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataError, EventViews
from .visibility import read_visible_window

FEATURES = ["age_rank", "log_mean", "cv", "positive_day_ratio"]


def profile_values(amounts, age, gender):
    values = np.asarray(amounts, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise DataError("INVALID_VALUE", "Profile requires finite nonnegative visible daily amounts")
    if age not in set("0123456U") or gender not in set("FMEU"):
        raise DataError("INVALID_VALUE", "Invalid profile attributes")
    mean = float(values.mean())
    sigma = float(values.std(ddof=0))
    return {"age": age, "gender": gender, "age_rank": None if age == "U" else int(age),
            "n": len(values), "mu": mean, "sigma": sigma, "log_mean": float(np.log1p(mean)),
            "cv": sigma / max(mean, 1.0), "positive_days": int((values > 0).sum()),
            "positive_day_ratio": float((values > 0).mean())}


class ProfileSource:
    """Customer slices reuse stage-one visibility without repeatedly copying the full log."""

    def __init__(self, events, panel):
        self.events = {customer: EventViews(rows.copy(), {}) for customer, rows in events.groupby("customer", sort=False)}
        self.panels = {customer: rows.copy() for customer, rows in panel.groupby("customer", sort=False)}

    def describe(self, customer, t, *, Lc=None, W=28):
        if customer not in self.events:
            raise DataError("NO_HISTORY", "Customer has no raw history", customer=customer)
        window = read_visible_window(self.events[customer], self.panels[customer], customer, t,
                                     Lc=Lc, require_labels=False)
        recent = window.history.tail(W)
        result = profile_values(recent.target_cents.to_numpy(dtype=np.float64) / 100,
                                window.attributes["age"], window.attributes["gender"])
        result.update(customer=customer, origin_t=t, history_start=window.history_start, Tobs=window.Tobs,
                      profile_start=int(recent.step.min()), **{key: window.attributes[key] for key in
                      ("attribute_source_step", "attribute_source_raw_row_id")})
        return result


def warm_profiles(source, train_ids, cfg):
    profiles = []
    warm_end = cfg["split"]["warm_end"]
    for customer in sorted(set(train_ids)):
        try:
            profile = source.describe(customer, warm_end, W=cfg["forecast"]["W"])
        except DataError as exc:
            if exc.code == "NO_HISTORY":
                continue
            raise
        if (profile["Tobs"] >= cfg["clustering"]["fit_min_visible_days"]
                and profile["gender"] in ("F", "M") and profile["age"] != "U"):
            profiles.append(profile)
    if not profiles:
        raise DataError("CLUSTER_FIT_FAILED", "No eligible warmup profiles")
    return pd.DataFrame(profiles)


def fit_amount_scale(panel, train_ids, warm_end=59):
    visible = panel.loc[panel.customer.isin(set(train_ids)) & panel.step.between(0, warm_end)
                        & panel.observed & panel.target_cents.gt(0), "target_cents"]
    return {"a": float(np.median(visible.to_numpy(dtype=np.float64)) / 100) if len(visible) else 1.0,
            "positive_daily_count": len(visible), "fit_start": 0, "fit_end": warm_end,
            "unit": "original_dataset_amount_unit", "fallback_no_positive_days": len(visible) == 0}


def apply_feature_weights(z, weights):
    return np.asarray(z, dtype=np.float64) * np.sqrt(np.asarray(weights, dtype=np.float64))


@dataclass
class ProfileTransform:
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    constant: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, profiles, weights=(0.6, 0.2, 0.1, 0.1)):
        x = profiles[FEATURES].to_numpy(dtype=np.float64, copy=True)
        if not len(x) or not np.isfinite(x).all():
            raise DataError("CLUSTER_FIT_FAILED", "Warmup profiles must be finite")
        lower, upper = np.quantile(x[:, [1, 2]], [0.01, 0.99], axis=0, method="linear")
        x[:, [1, 2]] = np.clip(x[:, [1, 2]], lower, upper)
        std = x.std(axis=0, ddof=0)
        constant = np.ptp(x, axis=0) == 0
        std[constant] = 0
        return cls(x.mean(axis=0), std, lower, upper, constant, np.asarray(weights, dtype=float))

    def transform(self, profiles):
        x = profiles[FEATURES].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(x).all():
            raise DataError("INVALID_VALUE", "Cannot transform unknown or nonfinite profile features")
        x[:, [1, 2]] = np.clip(x[:, [1, 2]], self.lower, self.upper)
        z = (x - self.mean) / np.where(self.constant, 1.0, self.std)
        z[:, self.constant] = 0
        return apply_feature_weights(z, self.weights)

    def to_dict(self):
        return {"feature_order": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "winsor_features": ["log_mean", "cv"], "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "weights": self.weights.tolist(), "constant_feature": self.constant.tolist(),
                "quantiles": [0.01, 0.99], "quantile_method": "linear", "std_ddof": 0}

    @classmethod
    def from_dict(cls, value):
        if value["feature_order"] != FEATURES:
            raise DataError("CONFIG_MISMATCH", "Saved feature order differs")
        return cls(*(np.asarray(value[key], dtype=bool if key == "constant_feature" else float)
                     for key in ("mean", "std", "lower", "upper", "constant_feature", "weights")))
