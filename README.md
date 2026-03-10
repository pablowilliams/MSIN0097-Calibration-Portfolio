# Calibration-Optimised Equity Outperformance Prediction

## Why This Project Matters

Most machine learning models in finance are evaluated on accuracy or AUC, but these metrics say nothing about whether a model's *confidence* is trustworthy. A model that predicts 70% probability of outperformance when the true rate is 55% might score well on accuracy, yet its overconfidence would lead a Kelly-sizing framework to commit six times more capital than warranted. In high-stakes portfolio allocation, this kind of probability error is far more damaging than getting a few binary labels wrong.

This project asks a simple question: **if you optimise a model for calibration quality (ECE) instead of accuracy, do the resulting probability estimates actually produce better portfolios?** The answer, it turns out, is nuanced. The calibration-optimised portfolio does outperform the accuracy-optimised one directionally, but the effect is not statistically significant over a four-year test window with near-random signal strength. The more interesting finding is that the calibration advantage appears to come from *how* the model was tuned, not from *what* features it was given.

---

## Project Overview

**Programme**: MSc Business Analytics, MSIN0097 Predictive Analytics, UCL School of Management  
**Universe**: 457 S&P 500 equities, 82,260 stock-month observations, January 2010 to December 2024  
**Primary Metric**: Expected Calibration Error (ECE, M = 10 equal-width bins)  
**Research Question**: Does optimising for ECE over accuracy produce better Kelly-sized portfolios?

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/pablowilliams/MSIN0097-Calibration-Portfolio.git
cd MSIN0097-Calibration-Portfolio
pip install -r requirements.txt

# Run the full pipeline (takes ~55 minutes due to yfinance downloads and Optuna tuning)
jupyter notebook PythonFinance.ipynb
```

**Important**: The notebook downloads live data from Yahoo Finance and FRED. If you re-run it, the numbers will differ from those in the report because stock prices update daily and Optuna's stochastic search produces different hyperparameters each run. The submitted notebook has all outputs saved from the final run.

---

## What I Did and Why

### The Problem

In equity markets, signal-to-noise ratios are extremely low. Most stocks outperform SPY roughly half the time (50.6% in this dataset), making accuracy nearly useless as a metric. But if a model can produce *well-calibrated* probabilities, even modest ones like 52-55%, then Kelly-style position sizing can exploit those edges systematically. The question is whether calibration quality translates into portfolio performance.

### The Approach

I built 20 features across four pillars, each capturing a different type of signal:

| Pillar | Count | What It Captures | Example Features |
|--------|-------|-----------------|-----------------|
| **Technical** | 8 | Price momentum and mean-reversion | 1/3/6/12-month returns, RSI, Bollinger bands |
| **Elo Ratings** | 4 | Persistent competitive advantage | Monthly Elo updates (K=20), win probability |
| **Bayesian** | 3 | Stabilised estimates via shrinkage toward sector means | James-Stein shrunk returns, shrinkage weight |
| **Macro** | 5 | Interest rate and volatility regime | Yield spread, VIX, unemployment, CPI |

I trained six models spanning three complexity levels, all tuned via Optuna TPE to minimise ECE rather than accuracy. The key comparison is between Random Forest (calibration-best, ECE = 0.026) and XGBoost (accuracy-best, 50.4%), since both are tree-based ensembles and the pairing isolates whether bagging versus boosting drives the calibration difference.

### The Temporal Design

Every decision in this project respects strict calendar ordering. No future information leaks into training.

```
Train                  Validation             Test (touched once)
2010-01 -------- 2017-12  |  2018-01 ---- 2020-12  |  2021-01 -------- 2024-12
   N = 43,872             |     N = 16,452          |     N = 21,936
                           |                         |
   Optuna 3-fold           |   Model selection       |   Portfolio simulation
   expanding-window CV     |   + ECE evaluation      |   + H1/H2 testing
```

### The Portfolio

Each month, I ranked stocks by predicted outperformance probability and selected the top quintile (~91 stocks) into an equal-weighted portfolio. I used equal weighting rather than Kelly-proportional sizing because the signal is so weak (AUC ~ 0.52) that Kelly fractions would amplify noise into extreme positions. Transaction costs of 5 basis points per leg are applied at each rebalance.

---

## Key Results

### Validation Performance (2018-2020)

| Model | ECE | Brier | AUC-ROC | Accuracy |
|-------|-----|-------|---------|----------|
| Dummy | 0.499 | 0.499 | 0.500 | 50.1% |
| Logistic Reg. | 0.040 | 0.251 | 0.519 | 50.2% |
| **Random Forest** | **0.026** | **0.250** | 0.524 | 50.1% |
| XGBoost | 0.031 | 0.251 | 0.525 | **50.4%** |
| LightGBM | 0.029 | 0.251 | 0.523 | 50.2% |
| MLP | 0.252 | 0.336 | 0.503 | 50.0% |

### Portfolio Performance (Test: 2021-2024)

| Strategy | Sharpe (ann.) | Total Return | Ann. Return | Max Drawdown |
|----------|--------------|-------------|-------------|--------------|
| **Portfolio B (RF)** | **0.77** | 68.1% | 13.9% | -20.0% |
| Portfolio A (XGB) | 0.71 | 63.4% | 13.1% | -17.4% |
| Equal Weight (1/N) | 0.82 | 68.5% | 13.9% | -18.4% |
| SPY Buy-and-Hold | 0.83 | 66.1% | 13.5% | -24.0% |

### Hypothesis Verdicts

**H1** (calibration-best beats accuracy-best): Directionally supported (Sharpe 0.77 > 0.71), but the paired t-test gives p = 0.294 with Cohen's d = 0.080 and a bootstrap CI [-0.080, 0.204] crossing zero. I cannot reject the null. The effect reverses during the 2022 bear market, suggesting any benefit is regime-dependent.

**H2** (engineered features improve calibration): Mixed. Removing Elo worsens ECE by 0.003, suggesting it captures genuine persistence. Removing Bayesian has negligible effect. Removing Macro actually *improves* ECE, likely because FRED's revised vintages introduce look-ahead bias. The full model (ECE = 0.026) barely outperforms the technical-only baseline (0.028), so the calibration advantage comes from ECE-targeted tuning rather than the novel features.

### What Went Wrong

The most damaging limitation is the 46% ECE degradation from validation (0.026) to test (0.038), driven by regime shift. The training period (2010-2017) was relatively calm, while the test period (2021-2024) included the post-COVID recovery, the 2022 rate-hiking drawdown, and the 2023-24 AI rally. Errors concentrate in the Energy sector, where geopolitical shocks had no historical analogue in training.

---

## Project Structure

```
MSIN0097-Calibration-Portfolio/
  PythonFinance.ipynb             # End-to-end pipeline (81 cells, ~55 min runtime)
  MSIN0097_Final_Report.pdf       # 2,000-word report with 20 embedded figures
  agent_collaboration_log.md      # AI-human collaboration audit (48 entries)
  claude.md                       # Agent runbook with temporal discipline rules
  requirements.txt                # Python 3.10+ dependencies
  .gitignore
  figures/                        # 28 auto-generated PNGs
  tables/                         # Auto-generated CSV outputs
  data/
    raw/                          # yfinance + FRED downloads
    processed/                    # Feature matrices, predictions
```

---

## Statistical Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Paired t-test (H1) | t(47) = 0.545 | 0.294 (one-sided) | Not significant |
| Cohen's d | 0.080 | -- | Below small-effect threshold (0.20) |
| Bootstrap 95% CI | [-0.080, 0.204] | -- | Crosses zero |
| Bonferroni (alpha = 0.025) | -- | -- | Does not survive |
| Shapiro-Wilk (normality) | W = 0.911 | 0.002 | Non-normal; t-test is approximate |

---

## Dependencies

Python 3.10+ with: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `matplotlib`, `seaborn`, `scipy`. Full list in `requirements.txt`.

---

## References

- Breiman, L. (2001) Random forests. *Machine Learning* 45(1), 5-32.
- Brier, G.W. (1950) Verification of forecasts. *Monthly Weather Review* 78(1), 1-3.
- Chen, T. and Guestrin, C. (2016) XGBoost. *KDD*, 785-794.
- Cont, R. (2001) Empirical properties of asset returns. *Quantitative Finance* 1(2), 223-236.
- DeMiguel, V. et al. (2009) Optimal versus naive diversification. *RFS* 22(5), 1915-1953.
- Efron, B. and Morris, C. (1973) Stein estimation. *JASA* 68(341), 117-130.
- Elo, A.E. (1978) *The Rating of Chessplayers*. Arco.
- Fama, E.F. (1970) Efficient capital markets. *JoF* 25(2), 383-417.
- Guo, C. et al. (2017) On calibration of modern neural networks. *ICML*, 1321-1330.
- Jegadeesh, N. and Titman, S. (1993) Returns to buying winners. *JoF* 48(1), 65-91.
- Kelly, J.L. (1956) A new interpretation of information rate. *BSTJ* 35(4), 917-926.
- Naeini, M.P. et al. (2015) Calibrated probabilities using Bayesian binning. *AAAI*.
- Widmann, D. et al. (2019) Calibration tests in multi-class classification. *NeurIPS*, 12257-12267.

---

*I used Claude (Anthropic) as a coding collaborator over three weeks. Every analytical decision, temporal safeguard, and domain judgement was mine. The agent wrote syntax; I caught the leakage, designed the temporal splits, chose the ECE objective, and diagnosed every failure. Full audit trail in `agent_collaboration_log.md`.*
