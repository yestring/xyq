"""Compact date indexes; materialize only one distance block at a time."""

from dataclasses import dataclass

import numpy as np

from .cluster_io import fingerprint
from .data import DataError


class HistoryStore:
    def __init__(self, panel):
        self.amounts, self.first = {}, {}
        for customer, rows in panel.groupby("customer", sort=True):
            if rows.step.duplicated().any() or rows.step.min() < 0:
                raise DataError("DATA_GAP", "Duplicate or negative panel days", customer=customer)
            values = np.full(int(rows.step.max()) + 1, np.nan)
            valid = rows.observed & rows.target_cents.notna()
            observed = rows.loc[valid]
            values[observed.step.to_numpy(dtype=int)] = observed.target_cents.to_numpy(dtype=float) / 100
            if (values[np.isfinite(values)] < 0).any():
                raise DataError("INVALID_VALUE", "Negative daily amount")
            values.flags.writeable = False
            self.amounts[customer] = values
            if len(observed):
                self.first[customer] = int(observed.step.min())


@dataclass
class CandidateBank:
    donors: tuple
    donor_index: np.ndarray
    key_end: np.ndarray
    ell: int
    H: int
    cutoff: int
    version: str
    groups: dict

    def __len__(self):
        return len(self.key_end)

    def sequences(self, store, indices, *, values=False):
        length = self.H if values else self.ell
        result = np.empty((len(indices), length), dtype=np.float64)
        for row, i in enumerate(indices):
            end = int(self.key_end[i])
            start = end + 1 if values else end - self.ell + 1
            result[row] = store.amounts[self.donors[self.donor_index[i]]][start:start + length]
        return result

    def record(self, i, experiment):
        donor = self.donors[self.donor_index[i]]
        end = int(self.key_end[i])
        start = end - self.ell + 1
        return {"candidate_id": f"{experiment}:{self.version}:{donor}:{start}:{end}:{end+self.H}",
                "donor_id": donor, "key_start": start, "key_end": end,
                "value_start": end + 1, "value_end": end + self.H,
                "ell": self.ell, "H": self.H, "donor_group": self.groups.get(donor),
                "source_cutoff": self.cutoff, "library_version": self.version}


def build_bank(store, donors, groups, *, ell, H, stride, cutoff, identity,
               allowed_start=None, value_limit=None):
    """Dates are anchored to each donor's allowed first day, including zero days."""
    donor_ids = tuple(sorted(donors))
    limit = cutoff if value_limit is None else min(cutoff, value_limit)
    donor_index, ends = [], []
    for j, donor in enumerate(donor_ids):
        if donor not in store.first:
            raise DataError("CONFIG_MISMATCH", "Donor is absent from observed panel", donor=donor)
        start = store.first[donor] if allowed_start is None else allowed_start
        values = store.amounts[donor]
        # Prefix count checks the entire K+V, not just endpoints.
        missing = np.r_[0, np.cumsum(~np.isfinite(values))]
        for end in range(start + ell - 1, min(limit, len(values) - 1) - H + 1, stride):
            key_start = end - ell + 1
            if missing[end + H + 1] != missing[key_start]:
                continue
            donor_index.append(j)
            ends.append(end)
    return CandidateBank(donor_ids, np.asarray(donor_index, dtype=np.int32),
                         np.asarray(ends, dtype=np.int32), ell, H, cutoff,
                         fingerprint(identity), dict(groups))


def exact_distances(bank, store, Q, indices, block_size):
    q = np.asarray(Q, dtype=np.float64)
    if q.shape != (bank.ell,) or not np.isfinite(q).all() or (q < 0).any():
        raise DataError("INVALID_VALUE", "Query must be finite, nonnegative and equal length")
    distances = np.empty(len(indices), dtype=np.float64)
    logq = np.log1p(q)
    peak = 0
    for start in range(0, len(indices), block_size):
        part = indices[start:start + block_size]
        keys = bank.sequences(store, part)
        distances[start:start + len(part)] = np.mean((np.log1p(keys) - logq) ** 2, axis=1)
        peak = max(peak, len(part))
    return distances, peak


def greedy_select(bank, indices, distances, *, m, cross, max_per_donor=2, group=None):
    # Lexicographic donor indexes correspond exactly to string donor_id order.
    order = np.lexsort((bank.key_end[indices], bank.donor_index[indices], distances))
    accepted, intervals = [], {}
    for position in order:
        i = int(indices[position])
        donor = bank.donors[bank.donor_index[i]]
        if group is not None and bank.groups[donor] != group:
            continue
        previous = intervals.setdefault(donor, [])
        start, end = int(bank.key_end[i]) - bank.ell + 1, int(bank.key_end[i]) + bank.H
        if cross and (len(previous) >= max_per_donor or any(start <= b and a <= end for a, b in previous)):
            continue
        accepted.append((i, float(distances[position])))
        previous.append((start, end))
        if len(accepted) == m:
            break
    return accepted


def reference_weights(distances, tau):
    if not np.isfinite(tau) or tau <= 0:
        raise DataError("INVALID_VALUE", "tau must be finite and positive")
    values = np.asarray(distances, dtype=np.float64)
    if not len(values):
        return np.empty(0)
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataError("INVALID_VALUE", "Invalid distances")
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(values - values.min()) / tau)
    return weights / weights.sum()
