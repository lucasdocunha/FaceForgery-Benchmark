from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

def mean(predictions, **_): return np.mean(np.asarray(predictions),axis=0)
def weighted_by_val_auc(predictions, weights, **_): return np.average(np.asarray(predictions),axis=0,weights=np.asarray(weights))
def majority_vote(predictions, **_): return (np.mean(np.asarray(predictions)>=.5,axis=0)>=.5).astype(float)
def max_prob(predictions, **_):
    p=np.asarray(predictions); return np.where(np.max(p,axis=0)>=1-np.min(p,axis=0),np.max(p,axis=0),np.min(p,axis=0))
def geometric_mean(predictions, **_): return np.exp(np.mean(np.log(np.clip(np.asarray(predictions),1e-9,1)),axis=0))
def stacking_logreg(predictions, y_true, test_predictions=None, **_):
    model=LogisticRegression().fit(np.asarray(predictions).T,y_true)
    target=predictions if test_predictions is None else test_predictions
    return model.predict_proba(np.asarray(target).T)[:,1]
STRATEGIES={"mean":mean,"weighted":weighted_by_val_auc,"majority":majority_vote,"max":max_prob,"geometric":geometric_mean,"stacking":stacking_logreg}
