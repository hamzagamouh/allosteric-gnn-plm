import random
import numpy as np
from sklearn.metrics import matthews_corrcoef


def evaluate_mcc(clf, X, y):
    return matthews_corrcoef(y.ravel(), clf.predict(X))


def drop_nan_rows(X, ref):
    return X[~np.isnan(ref).any(axis=1)]


def balance_subsample(allo_feats, non_allo_feats, seed=4652):
    """Downsample non-allosteric class to match allosteric class size."""
    random.seed(seed)
    selected = random.sample(range(len(non_allo_feats)), len(allo_feats))
    X = np.concatenate([allo_feats, non_allo_feats[selected]], axis=0)
    y = np.array([1] * len(allo_feats) + [0] * len(allo_feats)).reshape(-1, 1)
    return X, y


def build_val_arrays(allo_feats, non_allo_feats):
    X = np.concatenate([allo_feats, non_allo_feats], axis=0)
    y = np.array([1] * len(allo_feats) + [0] * len(non_allo_feats)).reshape(-1, 1)
    return X, y
