import numpy as np
from src.pipelines.ensemble_strategies import mean,weighted_by_val_auc,majority_vote,geometric_mean,stacking_logreg
def test_ensemble_strategies_preserve_sample_shape():
    p=np.array([[.1,.8,.2,.9],[.2,.7,.4,.8]]);y=np.array([0,1,0,1])
    assert mean(p).shape==weighted_by_val_auc(p,[1,2]).shape==majority_vote(p).shape==geometric_mean(p).shape==(4,)
    assert stacking_logreg(p,y).shape==(4,)
