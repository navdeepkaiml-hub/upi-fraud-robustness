import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import FEATURES


def _log1p_abs(X):
    return np.log1p(np.abs(X))


def make_models(seed=0):
    return {
        "LogReg": make_pipeline(FunctionTransformer(_log1p_abs), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=1000)),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=seed),
        "GradBoost": HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        "MLP": make_pipeline(FunctionTransformer(_log1p_abs), StandardScaler(),
                             MLPClassifier((64, 32), max_iter=200, early_stopping=True, random_state=seed)),
    }


def xy(df):
    return df[FEATURES].to_numpy(float), df["is_fraud"].to_numpy()


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.argmax(f1[:-1])])


def report(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    return dict(pr_auc=float(average_precision_score(y, p)),
                precision=tp / max(int(pred.sum()), 1), recall=tp / max(int((y == 1).sum()), 1))
