"""Weighted K-means, stable business naming, diagnostics and as-of assignment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from .allocation import NAMES, map_allocation
from .data import DataError
from .profiles import ProfileTransform, fit_amount_scale, warm_profiles


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            **dict(zip(("min", "q25", "median", "q75", "q99", "max"),
                       np.quantile(values, [0, .25, .5, .75, .99, 1], method="linear").tolist()))}


def name_clusters(profiles, labels):
    clusters = []
    for label in sorted(set(labels)):
        rows = profiles.loc[np.asarray(labels) == label]
        clusters.append((float(rows.age_rank.mean()), float(rows.mu.mean()), int(label)))
    if len(clusters) != 3:
        raise DataError("CLUSTER_FIT_FAILED", "Exactly three nonempty clusters required")
    return {str(label): {"business_code": code, "group_name": name}
            for (_, _, label), code, name in zip(sorted(clusters), ("A", "B", "C"), NAMES)}


def cluster_summary(profiles, labels, centers, group_map, cfg):
    summaries = []
    threshold = max(cfg["clustering"]["small_cluster_absolute"],
                    len(profiles) * cfg["clustering"]["small_cluster_fraction"])
    for label, mapping in sorted(group_map.items(), key=lambda item: item[1]["business_code"]):
        rows = profiles.loc[np.asarray(labels) == int(label)]
        counts = {str(age): int(rows.age_rank.eq(age).sum()) for age in range(7)}
        summaries.append({"raw_cluster": int(label), **mapping, "count": len(rows),
                          "fraction": len(rows) / len(profiles), "age_rank_mean": float(rows.age_rank.mean()),
                          "age_rank_counts": counts, "age_rank_proportions": {key: value / len(rows) for key, value in counts.items()},
                          "daily_mean_amount": distribution(rows.mu), "cv": distribution(rows.cv),
                          "positive_day_ratio": distribution(rows.positive_day_ratio),
                          "warm_all_zero_count": int(rows.mu.eq(0).sum()),
                          "center_weighted_standardized": np.asarray(centers)[int(label)].tolist(),
                          "small_cluster": len(rows) < threshold})
    gaps = np.diff([row["age_rank_mean"] for row in summaries]).tolist()
    semantic = any(gap < cfg["clustering"]["semantic_min_age_rank_gap"] for gap in gaps)
    return {"clusters": summaries, "small_cluster_threshold": threshold,
            "small_cluster": any(row["small_cluster"] for row in summaries),
            "semantic_warning": semantic, "adjacent_age_rank_gaps": gaps,
            "semantic_min_age_rank_gap": cfg["clustering"]["semantic_min_age_rank_gap"],
            "interpretation": "按年龄顺序映射，完整阶段画像尚未验证；仅描述年龄与正常消费。稳健投资比例不代表现金储备。",
            "age0_note": "年龄码0包括18岁及更年轻者，不能作为成年资格证明。"}


def fit_kmeans(profiles, transform, cfg, seed=42):
    weighted = transform.transform(profiles)
    if len(profiles) < 3 or len(np.unique(weighted, axis=0)) < 3:
        raise DataError("CLUSTER_FIT_FAILED", "Fewer than three distinct weighted feature vectors")
    params = cfg["clustering"]
    model = KMeans(n_clusters=3, init=params["init"], n_init=params["n_init"],
                   max_iter=params["max_iter"], tol=params["tol"], random_state=seed)
    with threadpool_limits(limits=1):
        model.fit(weighted)
    if len(np.unique(model.labels_)) != 3:
        raise DataError("CLUSTER_FIT_FAILED", "Fitted K-means has fewer than three nonempty clusters")
    mapping = name_clusters(profiles, model.labels_)
    summary = cluster_summary(profiles, model.labels_, model.cluster_centers_, mapping, cfg)
    summary.update(inertia=float(model.inertia_), iterations=int(model.n_iter_), seed=seed)
    return model, mapping, summary


@dataclass
class ClusterBundle:
    transform: ProfileTransform
    model: KMeans
    group_map: dict
    amount_scale: dict
    summary: dict
    metadata: dict


def fit_bundle(source, panel, train_ids, cfg, metadata):
    profiles = warm_profiles(source, train_ids, cfg)
    transform = ProfileTransform.fit(profiles, cfg["profile"]["weights"])
    scale = fit_amount_scale(panel, train_ids, cfg["split"]["warm_end"])
    model, mapping, summary = fit_kmeans(profiles, transform, cfg, cfg["clustering"]["random_state"])
    return ClusterBundle(transform, model, mapping, scale, summary, metadata), profiles


def stability_diagnostics(bundle, profiles, cfg):
    weighted = bundle.transform.transform(profiles)
    labels = {42: bundle.model.labels_.copy()}
    mappings = {42: bundle.group_map}
    runs = [{"seed": 42, "inertia": float(bundle.model.inertia_), "summary": bundle.summary}]
    for seed in (43, 44):
        model, mapping, summary = fit_kmeans(profiles, bundle.transform, cfg, seed)
        labels[seed], mappings[seed] = model.labels_.copy(), mapping
        runs.append({"seed": seed, "inertia": float(model.inertia_), "summary": summary})
    pairs = []
    for left, right in combinations(labels, 2):
        named_left = [mappings[left][str(int(label))]["business_code"] for label in labels[left]]
        named_right = [mappings[right][str(int(label))]["business_code"] for label in labels[right]]
        pairs.append({"seeds": [left, right], "adjusted_rand_index": float(adjusted_rand_score(labels[left], labels[right])),
                      "business_name_agreement": float(np.mean(np.asarray(named_left) == named_right))})
    with threadpool_limits(limits=1), config_context(working_memory=64):
        silhouette = float(silhouette_score(weighted, labels[42], metric="euclidean")) if len(profiles) > 3 else None
    return {"silhouette_weighted_space": silhouette, "silhouette_sample_count": len(profiles),
            "silhouette_method": "all eligible customers; exact Euclidean silhouette in weighted standardized coordinates",
            "same_cohort_and_transform": True, "runs": runs, "pairwise_stability": pairs}, labels


def empty_group(status, **details):
    result = {"raw_cluster": None, "business_code": None, "group_name": None, "group_status": status,
              "same_group_filter_allowed": False, "note": status, **details}
    if status in {"CLUSTER_FIT_FAILED", "CONFIG_MISMATCH", "DATA_GAP", "ATTRIBUTE_CONFLICT"}:
        result.update(prediction_status="BLOCKED", predicted_daily=None)
    return {**result, **map_allocation(result)}


def assign_profile(bundle, profile):
    if bundle is None:
        return empty_group("CLUSTER_FIT_FAILED")
    if profile["gender"] == "E":
        return empty_group("NON_PERSON")
    if profile["age"] == "U" or profile["gender"] == "U":
        return empty_group("GROUP_UNKNOWN")
    weighted = bundle.transform.transform(pd.DataFrame([profile]))[0]
    low_info = profile["n"] < 7 or profile["positive_days"] < 2
    if low_info:
        ordered = sorted(bundle.group_map, key=lambda key: bundle.group_map[key]["business_code"])
        label = int(min(ordered, key=lambda key: (weighted[0] - bundle.model.cluster_centers_[int(key), 0]) ** 2))
    else:
        label = int(np.argmin(np.sum((bundle.model.cluster_centers_ - weighted) ** 2, axis=1)))
    result = {"raw_cluster": label, **bundle.group_map[str(label)], "group_status": "AGE_ONLY" if low_info else "FULL",
              "same_group_filter_allowed": not low_info,
              "note": "仅按年龄暂定" if low_info else "按冻结的四维加权距离分群",
              "semantic_warning": bundle.summary["semantic_warning"]}
    return {**result, **map_allocation(result)}


def describe_customer(bundle, source, customer, t, *, Lc=None, W=28):
    context = {"customer": customer, "origin_t": t, "history_mode": "ordinary" if Lc is None else "cold", "Lc": Lc}
    if bundle is None:
        return {**context, **empty_group("CLUSTER_FIT_FAILED")}
    try:
        profile = source.describe(customer, t, Lc=Lc, W=W)
    except DataError as exc:
        return {**context, **empty_group(exc.code), "error": exc.as_dict()}
    return {**context, **assign_profile(bundle, profile), "profile": None if profile["gender"] == "E" else profile,
            "history_start": profile["history_start"], "Tobs": profile["Tobs"],
            "attribute_source_step": profile["attribute_source_step"],
            "attribute_source_raw_row_id": profile["attribute_source_raw_row_id"]}
