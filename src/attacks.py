"""Black-box evasion attack on a fraud detector.

Threat model: the attacker can query the model's score but not see its internals. For each fraud
transaction they may (a) split the amount into k smaller transfers (raising short-term velocity by k-1)
and (b) move the transaction into daytime hours. They cannot change payee novelty, device change
or account age. They try `n_queries` random strategies and keep the lowest-scoring one.
"""
import numpy as np

from .data import FEATURES

I = {f: i for i, f in enumerate(FEATURES)}


def evade(model, X_fraud, n_queries=40, max_split=6, seed=0):
    rng = np.random.default_rng(seed)
    best = X_fraud.copy()
    best_s = model.predict_proba(X_fraud)[:, 1]
    for _ in range(n_queries):
        cand = X_fraud.copy()
        k = rng.integers(1, max_split + 1, len(cand))
        cand[:, I["amount"]] = X_fraud[:, I["amount"]] / k
        cand[:, I["amount_ratio"]] = X_fraud[:, I["amount_ratio"]] / k
        cand[:, I["velocity_1h"]] = X_fraud[:, I["velocity_1h"]] + (k - 1)
        cand[:, I["hour"]] = rng.integers(9, 21, len(cand))
        s = model.predict_proba(cand)[:, 1]
        better = s < best_s
        best[better], best_s[better] = cand[better], s[better]
    return best
