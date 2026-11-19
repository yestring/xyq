"""Original-unit, customer-equal metrics and the five specified baselines."""

from collections import defaultdict

import numpy as np

from .data import DataError


def baselines(record, H=7):
    q = np.asarray(record["Q"],dtype=float)
    mean = float(q.mean())
    return {"zero":np.zeros(H), "last":np.full(H,q[-1]), "q_mean":np.full(H,mean),
            "q_median":np.full(H,np.median(q)),
            "retrieval":np.asarray(record["R"],dtype=float) if record["reference_available"] else np.full(H,mean)}


class MetricAccumulator:
    def __init__(self):
        self.rows = defaultdict(lambda: np.zeros(4,dtype=float))

    def add(self, customer, Lc, predicted, truth):
        predicted, truth = np.asarray(predicted,dtype=float),np.asarray(truth,dtype=float)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all() or not np.isfinite(truth).all():
            raise DataError("INVALID_VALUE", "Nonfinite/mismatched evaluation values")
        err = predicted-truth
        self.rows[(Lc or 0,customer)] += [np.mean(err**2),np.mean(abs(err)),abs(predicted.sum()-truth.sum()),1]

    def finish(self, experiment):
        lengths = [0] if experiment == "O" else [1,3,7,14]
        by_length = {}
        for length in lengths:
            records = [v for (lc,_),v in self.rows.items() if lc == length]
            if not records:
                raise DataError("VALIDATION_EMPTY", "An evaluation history stratum is empty", Lc=length)
            values = np.array([v[:3]/v[3] for v in records]).mean(axis=0)
            by_length[str(length)] = {"MSE":float(values[0]),"RMSE":float(np.sqrt(values[0])),
                                     "MAE":float(values[1]),"TotalAE":float(values[2]),
                                     "customers":len(records),"samples":int(sum(v[3] for v in records))}
        means = {metric:float(np.mean([m[metric] for m in by_length.values()])) for metric in ("MSE","RMSE","MAE","TotalAE")}
        return {"metrics":means,"by_Lc":by_length,"selection_mse":means["MSE"],
                "aggregation":"Within customer mean over origins/horizon, then customer equal; C reports equal mean of four Lc metrics"}


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience, self.best, self.best_epoch, self.bad_epochs = patience,float("inf"),None,0

    def update(self, score, epoch):
        if not np.isfinite(score):
            raise DataError("TRAINING_FAILED", "Nonfinite validation score")
        improved = score < self.best
        if improved:
            self.best,self.best_epoch,self.bad_epochs = float(score),epoch,0
        else:
            self.bad_epochs += 1
        return improved,self.bad_epochs >= self.patience
