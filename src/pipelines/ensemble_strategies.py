from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _array(predictions):
    value = np.asarray(predictions, dtype=float)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError("predictions must have shape [models, samples]")
    return np.clip(value, 0, 1)


def mean(predictions, **_):
    return _array(predictions).mean(axis=0)


def weighted_by_val_auc(predictions, weights, **_):
    values, weights = _array(predictions), np.asarray(weights, dtype=float)
    if len(weights) != len(values):
        raise ValueError("One validation AUC weight is required per model")
    weights = np.maximum(weights - .5, 0)
    return values.mean(axis=0) if weights.sum() == 0 else np.average(values, axis=0, weights=weights)


def majority_vote(predictions, **_):
    return ((_array(predictions) >= .5).mean(axis=0) >= .5).astype(float)


def max_prob(predictions, **_):
    return _array(predictions).max(axis=0)


def geometric_mean(predictions, **_):
    return np.exp(np.log(np.clip(_array(predictions), 1e-9, 1)).mean(axis=0))


def stacking_logreg(predictions, y_true, test_predictions=None, **_):
    model = LogisticRegression(random_state=42, max_iter=1000).fit(_array(predictions).T, y_true)
    target = predictions if test_predictions is None else test_predictions
    return model.predict_proba(_array(target).T)[:, 1]


STRATEGIES = {
    "mean": mean, "weighted": weighted_by_val_auc, "majority": majority_vote,
    "max": max_prob, "geometric": geometric_mean, "stacking": stacking_logreg,
}
