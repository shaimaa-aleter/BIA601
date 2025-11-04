import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from .config import CFG


def smart_k(n_features: int) -> int:
    if n_features <= 0:
        return 0
    if CFG.TOP_K_MODE.lower() == "fixed":
        return max(1, min(CFG.TOP_K, n_features))
    k = int(np.ceil(CFG.K_RATIO * n_features))
    return max(1, min(k, n_features))


def iqr_cap_series(s, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return s.clip(lower=lower, upper=upper)


def dynamic_cv(y: np.ndarray, desired_splits: int = None) -> int:
    if desired_splits is None:
        desired_splits = CFG.GA_CV
    counts = np.bincount(y.astype(int))
    min_class = counts.min() if counts.size > 0 else 1
    return int(max(2, min(5, desired_splits, min_class)))


def evaluate_subset(clf, X_tr, y_tr, X_te, y_te, use_proba=True) -> Dict:
    clf.fit(X_tr, y_tr)
    if use_proba and hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X_te)[:, 1]
        return {
            "accuracy": accuracy_score(y_te, clf.predict(X_te)),
            "roc_auc": roc_auc_score(y_te, p),
            "pr_auc": average_precision_score(y_te, p),
        }
    else:
        pred = clf.predict(X_te)
        return {
            "accuracy": accuracy_score(y_te, pred),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }