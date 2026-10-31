"""Runs all experiments end-to-end:  python run_experiments.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.attacks import evade
from src.data import generate, split
from src.models import best_f1_threshold, make_models, report, xy

os.makedirs("results", exist_ok=True)
df = generate()
S = split(df)
Xtr, ytr = xy(S["train"]); Xva, yva = xy(S["val"])
Xi, yi = xy(S["test_iid"]); Xs, ys = xy(S["test_shift"])
res = {"data": {k: dict(n=len(v), fraud=int(v.is_fraud.sum())) for k, v in S.items()}}
print(res["data"])

# ---------- Exp 1: baselines + distribution shift
models, thr = make_models(), {}
res["shift"] = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    thr[name] = best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
    res["shift"][name] = {"iid": report(yi, m.predict_proba(Xi)[:, 1], thr[name]),
                          "shifted": report(ys, m.predict_proba(Xs)[:, 1], thr[name])}
    print(name, {k: {a: round(b, 3) for a, b in v.items()} for k, v in res["shift"][name].items()})

# ---------- Exp 2: evasion attack (attacker queries the model, modifies fraud transactions)
res["evasion"] = {}
for name, m in models.items():
    Xf = Xi[yi == 1]
    adv = evade(m, Xf)
    rec_c = float((m.predict_proba(Xf)[:, 1] >= thr[name]).mean())
    rec_a = float((m.predict_proba(adv)[:, 1] >= thr[name]).mean())
    res["evasion"][name] = dict(recall_clean=rec_c, recall_attacked=rec_a)
    print("evasion", name, round(rec_c, 3), "->", round(rec_a, 3))

# ---------- Exp 3: defence = adversarial training
res["defence"] = {}
for name in ["GradBoost", "RandomForest"]:
    base = models[name]
    adv_tr = evade(base, Xtr[ytr == 1], seed=1)
    Xaug = np.vstack([Xtr, adv_tr]); yaug = np.concatenate([ytr, np.ones(len(adv_tr), int)])
    robust = make_models()[name].fit(Xaug, yaug)
    t2 = best_f1_threshold(yva, robust.predict_proba(Xva)[:, 1])
    Xf = Xi[yi == 1]
    adv_new = evade(robust, Xf, seed=2)  # adaptive: attacker re-attacks the NEW model
    res["defence"][name] = dict(
        clean_pr_auc_before=res["shift"][name]["iid"]["pr_auc"],
        clean_pr_auc_after=report(yi, robust.predict_proba(Xi)[:, 1], t2)["pr_auc"],
        recall_attacked_before=res["evasion"][name]["recall_attacked"],
        recall_attacked_after=float((robust.predict_proba(adv_new)[:, 1] >= t2).mean()),
        shifted_pr_auc_after=report(ys, robust.predict_proba(Xs)[:, 1], t2)["pr_auc"])
    print("defence", name, {k: round(v, 3) for k, v in res["defence"][name].items()})


# ---------- Exp 4: calibration + abstention (GradBoost)
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if mk.any():
            e += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    return float(e)


m = models["GradBoost"]
pi, ps = m.predict_proba(Xi)[:, 1], m.predict_proba(Xs)[:, 1]
res["calibration"] = {"ece_iid": ece(yi, pi), "ece_shifted": ece(ys, ps)}
lo, hi = 0.2, 0.8
res["abstention"] = {}
for tag, y, p in [("iid", yi, pi), ("shifted", ys, ps)]:
    auto = (p <= lo) | (p >= hi)
    pred = p >= 0.5
    res["abstention"][tag] = dict(
        sent_to_human=float(1 - auto.mean()),
        error_rate_all=float((pred != y).mean()),
        error_rate_auto_only=float((pred[auto] != y[auto]).mean()),
        fraud_recall_if_human_reviews_uncertain=float(((p >= lo) & (y == 1)).sum() / max((y == 1).sum(), 1)))
print("calibration", res["calibration"]); print("abstention", res["abstention"])

# ---------- plots
names = list(models); w = 0.38; x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(x - w/2, [res["shift"][n]["iid"]["pr_auc"] for n in names], w, label="pre-drift test")
ax.bar(x + w/2, [res["shift"][n]["shifted"]["pr_auc"] for n in names], w, label="post-drift test")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("PR-AUC"); ax.legend(); ax.set_title("Distribution shift")
fig.tight_layout(); fig.savefig("results/shift_pr_auc.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(x - w/2, [res["evasion"][n]["recall_clean"] for n in names], w, label="clean")
ax.bar(x + w/2, [res["evasion"][n]["recall_attacked"] for n in names], w, label="after evasion attack")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("fraud recall"); ax.legend(); ax.set_title("Evasion attack")
fig.tight_layout(); fig.savefig("results/evasion_recall.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for tag, y, p in [("pre-drift", yi, pi), ("post-drift", ys, ps)]:
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, marker="o", label=tag)
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed fraud rate")
ax.legend(); ax.set_title("Reliability (GradBoost)")
fig.tight_layout(); fig.savefig("results/calibration.png", dpi=150); plt.close(fig)

json.dump(res, open("results/metrics.json", "w"), indent=2)
print("done")
