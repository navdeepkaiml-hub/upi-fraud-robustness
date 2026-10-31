"""Synthetic UPI-style transaction generator with a deliberate fraud-pattern drift.

Days 0-20  : "old" regime (fraud = large amounts, night-time, bursts, new payees).
Days 20-30 : "drift" regime (fraudsters adapt: smaller amounts, daytime, fewer bursts;
             legit users also change behaviour, e.g. festive-season spending).
This is SYNTHETIC data. Conclusions are about method behaviour, not real UPI fraud rates.
"""
import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "velocity_1h", "new_payee", "device_change",
            "account_age_days", "amount_ratio"]


def _hours(rng, n, night_prob):
    night = rng.random(n) < night_prob
    return np.where(night, rng.integers(0, 6, n), rng.integers(7, 23, n))


def generate(n=60000, fraud_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 30, n))
    late = t >= 20
    y = (rng.random(n) < fraud_rate).astype(int)
    stealth = (rng.random(n) < 0.25) & (y == 1)  # fraud that mimics legit behaviour

    # ---- legit behaviour (mild shift in the late regime)
    amt = rng.lognormal(np.where(late, 6.2, 6.0), 1.0, n)
    hour = _hours(rng, n, 0.04)
    vel = rng.poisson(0.3, n).astype(float)
    new_payee = (rng.random(n) < np.where(late, 0.16, 0.10)).astype(int)
    dev = (rng.random(n) < 0.02).astype(int)
    age = rng.exponential(400, n)
    avg30 = rng.lognormal(6.0, 0.6, n)

    # ---- fraud behaviour overwrites rows where y == 1 and not stealthy
    f = (y == 1) & ~stealth
    k = f.sum()
    fl = late[f]
    amt[f] = rng.lognormal(np.where(fl, 7.0, 8.0), 0.8, k)
    hour[f] = _hours(rng, k, np.where(fl, 0.20, 0.55))
    vel[f] = rng.poisson(np.where(fl, 1.0, 2.5), k)
    new_payee[f] = (rng.random(k) < np.where(fl, 0.55, 0.85)).astype(int)
    dev[f] = (rng.random(k) < 0.35).astype(int)
    age[f] = rng.exponential(120, k)

    return pd.DataFrame(dict(t=t, amount=amt, hour=hour, velocity_1h=vel,
                             new_payee=new_payee, device_change=dev,
                             account_age_days=age, amount_ratio=amt / avg30, is_fraud=y))


def split(df):
    """Temporal split. train < day 14, val 14-17, iid test 17-20, shifted test >= 20."""
    return dict(train=df[df.t < 14], val=df[(df.t >= 14) & (df.t < 17)],
                test_iid=df[(df.t >= 17) & (df.t < 20)], test_shift=df[df.t >= 20])
