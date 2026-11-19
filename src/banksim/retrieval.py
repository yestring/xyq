"""Frozen dual-path retrieval. No prediction, training, or cluster refitting."""

from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .candidates import HistoryStore, build_bank, exact_distances, greedy_select, reference_weights
from .cluster_io import fingerprint
from .clustering import assign_profile
from .data import DataError
from .profiles import profile_values
from .visibility import read_visible_window


@dataclass(frozen=True)
class RetrievalParams:
    H: int = 7
    L: int = 28
    C: int = 14
    W: int = 28
    m: int = 5
    tau: float = 1.0
    stride: int = 7
    max_per_donor: int = 2
    eval_cutoff: int = 119
    block_size: int = 2048

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "tau":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "eval_cutoff" else 1):
                raise DataError("INVALID_VALUE", "Invalid retrieval parameter", parameter=key)
        reference_weights([], self.tau)


class RetrievalEngine:
    def __init__(self, experiment, source, bundle, donor_groups, train_ids, versions,
                 *, params=None, cache=None, store=None):
        if bundle is None:
            raise DataError("CLUSTER_FIT_FAILED", "Retrieval requires accepted frozen clustering")
        if experiment not in ("O", "C", "TOY", "TEST"):
            raise DataError("INVALID_VALUE", "Unknown experiment")
        if not set(donor_groups).issubset(set(train_ids)) or not set(donor_groups).issubset(source.panels):
            raise DataError("CONFIG_MISMATCH", "Donors must belong to this experiment's training cohort")
        if any(str(group) not in bundle.group_map for group in donor_groups.values()):
            raise DataError("CONFIG_MISMATCH", "Unknown frozen donor cluster")
        self.experiment, self.source, self.bundle = experiment, source, bundle
        self.donor_groups = dict(donor_groups)
        self.params = params or RetrievalParams()
        self.versions = {**versions, "donor_labels_hash": fingerprint(donor_groups),
                         "training_ids_hash": fingerprint(sorted(train_ids))}
        self.store = store or HistoryStore(pd.concat(source.panels.values(), ignore_index=True))
        self.cache = cache
        self._banks = OrderedDict()

    def retrieve(self, customer, t, *, Lc=None, partition="train", method="same_group", library_mode="frozen"):
        p = self.params
        if partition not in ("train", "validation", "test") or method not in ("same_group", "all") or library_mode not in ("frozen", "rolling"):
            raise DataError("INVALID_VALUE", "Invalid retrieval request mode")
        if not isinstance(customer, str) or not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise DataError("INVALID_VALUE", "Customer must be a string; origin a nonnegative integer")
        if Lc is not None and (not isinstance(Lc, int) or isinstance(Lc, bool) or Lc < 1):
            raise DataError("INVALID_VALUE", "Lc must be a positive integer")
        cutoff = t if partition == "train" or library_mode == "rolling" else p.eval_cutoff
        if cutoff > t:
            raise DataError("INVALID_VALUE", "Candidate cutoff exceeds query origin")
        context = {"experiment": self.experiment, "customer": customer, "origin_t": t,
                   "history_mode": "ordinary" if Lc is None else "cold", "Lc": Lc,
                   "partition": partition, "method": method, "library_mode": library_mode,
                   "candidate_cutoff": cutoff, "parameters": asdict(p), "versions": self.versions}
        result = {**context, "cache_hit": False, "Q": [], "Tobs": 0, "ell": 0, "H": p.H,
                  "history_start": None, "group_status": "NO_HISTORY", "group_name": None,
                  "retrieval_path": "NONE", "retrieval_status": "NO_HISTORY", "selection_status": None,
                  "expanded_all": False, "fallback_reason": [], "R": [0.0] * p.H,
                  "r_log": [0.0] * p.H, "r_mask": 0, "reference_available": 0,
                  "candidate_count": 0, "candidate_ids": [], "candidates": [],
                  "scored_candidate_count": 0, "max_distance_block_rows": 0}
        try:
            if customer not in self.source.events:
                raise DataError("NO_HISTORY", "No raw history")
            window = read_visible_window(self.source.events[customer], self.source.panels[customer],
                                         customer, t, Lc=Lc, H=p.H, require_labels=False)
        except DataError as exc:
            if exc.code != "NO_HISTORY":
                raise
            result["fallback_reason"] = ["NO_HISTORY"]
            return self._finish(context, result)
        ell = min(p.L, window.Tobs)
        Q = window.history.tail(ell).target_cents.to_numpy(dtype=float) / 100
        recent = window.history.tail(p.W)
        profile = profile_values(recent.target_cents.to_numpy(dtype=float) / 100,
                                 window.attributes["age"], window.attributes["gender"])
        group = assign_profile(self.bundle, profile)
        cross = window.Tobs <= p.C
        result.update(Q=Q.tolist(), Tobs=window.Tobs, ell=ell, history_start=window.history_start,
                      query_start=t-ell+1, query_end=t, profile=profile, attributes=window.attributes,
                      group_status=group["group_status"], group_name=group["group_name"],
                      raw_cluster=group.get("raw_cluster"), retrieval_path="CROSS" if cross else "SELF")
        key = {**context, "history_start": window.history_start, "Tobs": window.Tobs, "ell": ell,
               "visible_query_hash": fingerprint({"Q": Q.tolist(), "profile": profile, "attributes": window.attributes})}
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return {**cached, "cache_hit": True}
        bank_key = {"experiment": self.experiment, "versions": self.versions, "ell": ell, "H": p.H,
                    "stride": p.stride, "cutoff": cutoff, "library_mode": library_mode,
                    "route": result["retrieval_path"], "self_customer": None if cross else customer,
                    "allowed_start": None if cross else window.history_start,
                    "value_limit": cutoff if cross else min(cutoff, t-ell)}
        bank_id = fingerprint(bank_key)
        if bank_id not in self._banks:
            def factory():
                return build_bank(self.store, self.donor_groups if cross else [customer], self.donor_groups,
                                  ell=ell, H=p.H, stride=p.stride, cutoff=cutoff, identity=bank_key,
                                  allowed_start=bank_key["allowed_start"], value_limit=bank_key["value_limit"])
            self._banks[bank_id] = self.cache.bank(bank_key, factory) if self.cache else factory()
            if len(self._banks) > 8:
                self._banks.popitem(last=False)
        bank = self._banks[bank_id]
        self._banks.move_to_end(bank_id)
        indices = np.arange(len(bank))
        if cross and customer in bank.donors:
            indices = indices[bank.donor_index != bank.donors.index(customer)]
        distances, peak = exact_distances(bank, self.store, Q, indices, p.block_size)
        select = dict(m=p.m, cross=cross, max_per_donor=p.max_per_donor)
        same = cross and method == "same_group" and group["group_status"] == "FULL"
        accepted = greedy_select(bank, indices, distances, **select,
                                 group=group["raw_cluster"] if same else None)
        result["selection_status"] = "SAME_GROUP" if same else "ALL_DONORS" if cross else "SELF"
        if same and len(accepted) < p.m:
            result.update(expanded_all=True, same_group_accepted_count=len(accepted), selection_status="EXPANDED_ALL")
            result["fallback_reason"].append("SAME_GROUP_ACCEPTED_BELOW_M")
            accepted = greedy_select(bank, indices, distances, **select)
        elif cross and not same:
            result["fallback_reason"].append("ALL_LIBRARY_CONTROL" if method == "all" else group["group_status"])
        weights = reference_weights([distance for _, distance in accepted], p.tau)
        selected = [i for i, _ in accepted]
        values = bank.sequences(self.store, selected, values=True)
        reference = np.sum(weights[:, None] * values, axis=0) if selected else np.zeros(p.H)
        records = []
        for position, ((i, distance), weight) in enumerate(zip(accepted, weights)):
            records.append({**bank.record(i, self.experiment), "distance": distance, "weight": float(weight),
                            "K": bank.sequences(self.store, [i])[0].tolist(), "V": values[position].tolist()})
        if len(selected) < p.m:
            result["fallback_reason"].append("NO_LEGAL_CANDIDATE" if not selected else "AVAILABLE_BELOW_M")
        result.update(R=reference.tolist(), r_log=np.log1p(reference).tolist(), r_mask=int(bool(selected)),
                      reference_available=int(bool(selected)), candidate_count=len(selected), candidates=records,
                      candidate_ids=[r["candidate_id"] for r in records], library_version=bank.version,
                      retrieval_status=result["selection_status"] if selected else "NO_REFERENCE",
                      scored_candidate_count=len(indices), max_distance_block_rows=peak)
        return self._finish(key, result)

    def _finish(self, key, result):
        result["cache_key"] = fingerprint(key)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return {**cached, "cache_hit": True}
            self.cache.put(key, result)
        return result
