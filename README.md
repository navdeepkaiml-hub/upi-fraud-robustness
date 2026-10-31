# Robustness of ML fraud detectors for UPI-style payments

**Research question:** How reliable are ML fraud detectors when (1) fraud patterns drift over time and (2) an attacker adapts to evade ML fraud detectors and what cheap mitigations help?

I write this as a fully reproducible study motivated by AI-safety themes: robustness to distribution shift, adversarial pressure, calibration and deferring to humans when a ML fraud detectors model is uncertain.

**Important:** all data is **synthetic** (see `src/data.py`). Results show how the *methods* behave under controlled conditions, not UPI fraud rates. Single run (seed 0); test sets contain 117 (pre-drift) and 413 (post-drift) fraud cases so differences of a few points are within noise.

https://github.com/user-attachments/assets/dfbbe734-e179-4788-874a-fa2a78617356

## Setup

- 60k transactions over 30 days 2% fraud. Features: amount, hour 1h velocity, new payee, device change, account age, amount ratio.

- split: train days 0-14 validation 14-17 **pre-drift test** 17-20 **post-drift test** 20-30.

- At day 20 fraudsters change behaviour ( amounts, more daytime, fewer bursts) and legit behaviour shifts slightly.

- Models: Logistic Regression, Random Forest, Gradient Boosting, MLP (scikit-learn). Threshold = F1 on validation.

- Attack (black-box evasion, `src/attacks.py`): attacker queries the score and may split an amount into k transfers and shift the time into daytime; keeps the lowest-scoring variant out of 40 tries. Cannot change payee/device/account age.

## Results
**1. Distribution shift (PR-AUC / recall)**

| Model | Pre-drift PR-AUC | Post-drift PR-AUC | Pre-drift recall | Post-drift recall |
|---|---|---|---|---|
| LogReg | 0.731 | 0.296 | 0.624 | 0.235 |
| RandomForest | 0.747 | 0.239 | 0.701 | 0.235 |
| GradBoost | 0.753 | 0.251 | 0.658 | 0.218 |
| MLP | 0.741 | 0.281 | 0.658 | 0.220 |

Every model loses roughly two thirds of its PR-AUC after drift; nothing in the metrics on old data warns about this.

**2. Evasion attack (fraud recall, pre-drift test)**

| Model | Clean | Attacked |
|---|---|---|
| LogReg | 0.624 | 0.538 |
| RandomForest | 0.701 | 0.368 |
| GradBoost | 0.658 | 0.444 |
| MLP | 0.658 | 0.479 |

Tree ensembles had the best clean recall but lost the most under attack; the simple linear model degraded least.

**3. Defence: adversarial training** (attacked fraud samples added to training; attacker then re-attacks the new model)

| Model | Clean PR-AUC before → after | Recall under attack before → after |
|---|---|---|
| GradBoost | 0.753 → 0.722 | 0.444 → 0.453 |
| RandomForest | 0.747 → 0.729 | 0.368 → 0.436 |

Gains are small and inconsistent (none for GradBoost) at a small cost in clean PR-AUC. This simple defence is not sufficient.

**4. Calibration and deferring to humans (GradBoost)**

- Expected calibration error: 0.098 pre-drift vs 0.105 post-drift (class-weighting already makes probabilities poorly calibrated).

- Sending predictions with probability in [0.2 0.8] to review (12% of cases pre-drift 14% post-drift): fraud recall, with human review of the uncertain band is **0.84 pre-drift but only 0.54 post-drift**. Under drift many frauds get *confident* scores, so confidence-based abstention gives much weaker protection than it appears to.

Plots: `results/shift_pr_auc.png` `results/evasion_recall.png` `results/calibration.png`. Raw numbers: `results/metrics.json`.

## Limitations

- Synthetic data and a hand-designed attacker; real fraud and real adversaries are messier.

- One seed, fraud counts, no confidence intervals.

- Attack is feature-space only; real attackers must produce valid transactions.

- Calibration was not corrected (no isotonic/Platt scaling) and no drift detector was evaluated.

## Next steps

Use seeds while showing confidence intervals. Work with semi‑real data such, as PaySim. Calibrate the models. Detect drift by using PSI or a classifier two‑sample test. Build attacks and stronger defences.

## Reproduce

```bash

pip install -r requirements.txt

python run_experiments.py      # -2 minutes writes results/

```
