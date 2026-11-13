# Equalized-odds post-processing (Hardt, Price & Srebro, 2016)

This repository evaluates the post-processing method from *Equality of Opportunity in Supervised Learning* (NeurIPS 2016) on the Adult, German Credit and COMPAS datasets. It uses Fairlearn's `ThresholdOptimizer(constraints="equalized_odds")`. The code is in `equalized_odds_evaluation.ipynb`, which is saved with all outputs.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook equalized_odds_evaluation.ipynb
```

Then choose **Kernel → Restart & Run All** (about 30 seconds). The datasets download automatically, so an internet connection is needed. Tested with Python 3.13, fairlearn 0.15, scikit-learn 1.9 and pandas 3.0.

## Data

| Dataset | Rows | Protected attribute (privileged group) | Favourable label |
|---|---|---|---|
| Adult (UCI) | 45,222 | sex (male) | income > 50K |
| German Credit (UCI) | 1,000 | age (> 25) | good credit risk |
| COMPAS (ProPublica) | 5,278 | race (Caucasian) | no recidivism within 2 years |

## Procedure and parameters

1. Compute DP_Diff, DIR and consistency on the original labels.
2. Split the data 70/30 into train and test, stratified on label × group.
3. Train `LogisticRegression(max_iter=2000)` on 70% of the training set.
4. Fit `ThresholdOptimizer(constraints="equalized_odds", prefit=True, predict_method="predict_proba")` on the remaining 30% of the training set.
5. Evaluate both models on the test set: DP_Diff, DIR, consistency (k = 5), equalized-odds gap, accuracy and balanced accuracy.

Steps 2–5 are repeated for `SEEDS = 10` random splits, and results are reported as mean ± std. `SEEDS` and `K` are set in the first code cell.

## Main results (test set, mean over 10 seeds)

| Dataset | EOdds gap: baseline → post-processed | DIR: baseline → post-processed | Accuracy: baseline → post-processed |
|---|---|---|---|
| Adult | 0.114 → 0.022 | 0.289 → 0.562 | 0.846 → 0.830 |
| German | 0.138 → 0.082 | 0.804 → 0.943 | 0.748 → 0.727 |
| COMPAS | 0.304 → 0.058 | 0.628 → 0.942 | 0.674 → 0.643 |

Post-processing removes most of the equalized-odds gap on Adult and COMPAS at a cost of 1.6–3.1 accuracy points. Results on the small German dataset are unstable.

