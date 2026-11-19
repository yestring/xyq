"""Admission from saved stage-two samples, never from a panel-wide random draw."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster_io import fingerprint
from .data import DataError, sha256_file
from .retrieval_io import read_json


@dataclass(frozen=True)
class SampleKey:
    customer: str
    t: int
    Lc: int | None = None


class LegalSamples:
    def __init__(self, engine, partition, frame=None):
        if partition not in ("train", "validation", "test"):
            raise DataError("INVALID_VALUE", "Unknown sample partition")
        self.engine, self.partition = engine, partition
        self.experiment = engine.experiment
        self.path = Path(f"artifacts/stage2/full/{self.experiment}/samples_{partition}.parquet")
        self.split = read_json(f"artifacts/splits/{self.experiment}.json")
        self.source_hash = sha256_file(self.path)
        self.frame = pd.read_parquet(self.path) if frame is None else frame.copy()
        if "source_row" not in self.frame:
            self.frame["source_row"] = np.arange(len(self.frame))
        self.frame = self.frame.reset_index(drop=True)
        self.frame["lc_key"] = self.frame.Lc.fillna(0).astype(int)
        self.validate()
        self.index = pd.MultiIndex.from_frame(self.frame[["customer", "origin_t", "lc_key"]])
        self.identity = fingerprint({"experiment": self.experiment, "partition": partition,
                                     "source_hash": self.source_hash, "source_rows": self.frame.source_row.tolist()})
        self.ordinary = self.frame.loc[self.frame.history_mode.eq("ordinary")]
        self.customer_origins = {c: g.origin_t.to_numpy(dtype=int) for c,g in self.ordinary.groupby("customer", sort=True)}
        self.customers = sorted(self.customer_origins)
        self.cold_options = {(c,int(t)): g.lc_key.to_numpy(dtype=int) for (c,t),g in
                             self.frame.loc[self.frame.lc_key.gt(0)].groupby(["customer", "origin_t"], sort=False)}

    def validate(self):
        e, f, part = self.engine, self.frame, self.partition
        split = self.split["ids"]
        if (e.bundle.metadata["fit_cutoff"] != 59 or e.bundle.metadata["fit_start"] != 0
                or e.bundle.metadata["experiment"] != self.experiment
                or e.bundle.metadata["data_hash"] != e.versions["data_hash"]
                or e.bundle.amount_scale["fit_start"] != 0 or e.bundle.amount_scale["fit_end"] != 59):
            raise DataError("CONFIG_MISMATCH", "Preprocessing/clustering fit lineage is incompatible")
        if not set(e.donor_groups) <= set(split["train"]):
            raise DataError("CONFIG_MISMATCH", "Donors are not training IDs")
        if self.experiment == "C":
            if any(set(split[a]) & set(split[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
                raise DataError("CONFIG_MISMATCH", "Strict split overlap")
            if set(e.donor_groups) & (set(split["validation"]) | set(split["test"])):
                raise DataError("CONFIG_MISMATCH", "Held-out donor contamination")
        if not set(f.customer) <= set(split[part]) or not f.experiment.eq(self.experiment).all() or not f.partition.eq(part).all():
            raise DataError("INVALID_VALUE", "Sample customer/experiment/partition mismatch")
        bounds = {"train": (59,112,119), "validation": (119,142,149), "test": (149,172,179)}[part]
        required = ["origin_t","target_start","target_end","H","history_start","history_end",
                    "attribute_start","attribute_end","Tobs","ell"]
        if f[required].isna().any().any() or any(not np.equal(f[k],np.floor(f[k])).all() for k in required):
            raise DataError("INVALID_VALUE", "Sample dates and lengths must be non-null integers")
        first = f.customer.map(e.store.first)
        ordinary = f.history_mode.eq("ordinary")
        cold = f.history_mode.eq("cold")
        checks = [f.origin_t.between(bounds[0],bounds[1]), f.target_start.eq(f.origin_t+1),
                  f.target_end.eq(f.origin_t+7), f.target_end.le(bounds[2]), f.H.eq(7),
                  f.history_end.eq(f.origin_t), f.history_start.ge(first),
                  f.attribute_start.eq(f.history_start), f.attribute_end.eq(f.origin_t),
                  f.Tobs.eq(f.origin_t-f.history_start+1), f.Tobs.gt(0), f.ell.eq(np.minimum(28,f.Tobs)),
                  ordinary | cold, (~ordinary) | (f.Lc.isna() & f.history_start.eq(first)),
                  (~cold) | (f.lc_key.isin([1,3,7,14]) & f.Tobs.eq(f.lc_key))]
        if any(not condition.all() for condition in checks) or f.duplicated(["customer","origin_t","lc_key"]).any():
            raise DataError("INVALID_VALUE", "Illegal origin/history/label window")
        if part != "train" and not (ordinary.all() if self.experiment == "O" else cold.all()):
            raise DataError("INVALID_VALUE", "Evaluation history mode differs from protocol")
        # Recheck complete observation over history AND labels, independently of persisted flags.
        for customer, rows in f.groupby("customer", sort=False):
            x = e.store.amounts[customer]
            missing = np.r_[0,np.cumsum(~np.isfinite(x))]
            start = rows.history_start.to_numpy(dtype=int)
            end = rows.target_end.to_numpy(dtype=int)
            if (end >= len(x)).any() or np.any(missing[end+1] != missing[start]):
                raise DataError("DATA_GAP", "Admitted sample contains unobserved history or labels")

    def subset(self, customers, origins):
        f = self.frame.loc[self.frame.customer.isin(customers) & self.frame.origin_t.isin(origins)]
        return LegalSamples(self.engine, self.partition, f)

    def locate(self, key):
        try:
            i = int(self.index.get_loc((key.customer, key.t, key.Lc or 0)))
        except KeyError as exc:
            raise DataError("INVALID_VALUE", "Request is not a saved legal sample in this partition",
                            customer=key.customer, origin_t=key.t, Lc=key.Lc, partition=self.partition) from exc
        row = self.frame.iloc[i]
        x = self.engine.store.amounts[key.customer]
        if not np.isfinite(x[int(row.history_start):int(row.target_end)+1]).all():
            raise DataError("DATA_GAP", "Sample changed after admission")
        return i, row

    def key_at(self, i):
        r = self.frame.iloc[i]
        return SampleKey(r.customer, int(r.origin_t), int(r.lc_key) or None)

    def target_cents(self, key):
        # Cache exact accepted integer cents, with -1 for unobserved cells; never reconstruct cents from floats.
        e = self.engine
        if not hasattr(e,"_forecast_cents"):
            e._forecast_cents = {}
        if key.customer not in e._forecast_cents:
            panel = e.source.panels[key.customer]
            observed = panel.loc[panel.observed & panel.target_cents.notna()]
            cents = np.full(len(e.store.amounts[key.customer]),-1,dtype=np.int64)
            cents[observed.step.to_numpy(dtype=int)] = observed.target_cents.to_numpy(dtype=np.int64)
            cents.flags.writeable = False
            e._forecast_cents[key.customer] = cents
        target = e._forecast_cents[key.customer][key.t+1:key.t+8]
        if len(target) != 7 or (target < 0).any():
            raise DataError("DATA_GAP", "Target cents are not fully observed")
        return target


class CustomerSampler:
    def __init__(self, catalog, seed):
        if catalog.partition != "train" or not catalog.customers:
            raise DataError("INVALID_VALUE", "Sampler needs admitted ordinary training windows")
        self.catalog = catalog
        self.rng = np.random.default_rng(seed)
        self.no_feasible_cold = 0

    def epoch_size(self, batch_size):
        return ((len(self.catalog.ordinary)+batch_size-1)//batch_size)*batch_size

    def draw(self):
        c = self.catalog.customers[int(self.rng.integers(len(self.catalog.customers)))]
        origins = self.catalog.customer_origins[c]
        t = int(origins[int(self.rng.integers(len(origins)))])
        Lc = None
        if self.rng.random() < .5:
            options = self.catalog.cold_options.get((c,t), [])
            if len(options):
                Lc = int(options[int(self.rng.integers(len(options)))])
            else:
                self.no_feasible_cold += 1
        return SampleKey(c,t,Lc)
