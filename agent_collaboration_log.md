# AI-Human Collaboration Audit Log

**Project**: Calibration-Optimised Equity Outperformance Prediction  
**Programme**: MSc Business Analytics (MSIN0097), UCL School of Management  
**AI Tool**: Claude (Anthropic) via claude.ai  
**Date Range**: February - March 2026  
**Total Entries**: 48  

---

## Purpose

This document provides a transparent, auditable record of every significant AI interaction throughout this project  -  what was delegated, what was rejected, and what was corrected. It demonstrates the plan-delegate-verify-revise cycle required by the MSIN0097 brief and evidences critical evaluation of all AI outputs.

---

## Section 1: Task Decomposition & Decision Register

| # | Task | Led By | Human Contribution | AI Contribution | Date |
|---|------|--------|--------------------|-----------------|------|
| 1 | **Research question formulation** | Human | Defined the calibration-vs-accuracy framing. Chose Kelly Criterion as the economic motivation linking calibration to portfolio outcomes. | Suggested binary outperformance as the target variable. | 2026-02-15 |
| 2 | **Data source selection** | Human | Selected Yahoo Finance for equity prices, FRED for macro variables. Identified the S&P 500 universe as the investment scope. | N/A  -  data procurement was entirely manual. | 2026-02-15 |
| 3 | **Feature engineering design** | Collaborative | Designed the four-pillar architecture (Technical, Elo, Bayesian, Macro). Chose K=20 for Elo based on FIDE midpoint reasoning. Specified which FRED variables to include. | Generated the Elo update loop and Bayesian shrinkage code. I verified that Elo ratings started at 1500 and diverged sensibly. | 2026-02-16 |
| 4 | **Temporal split design** | Human | Insisted on strict calendar-order splits with no shuffling. Defined the 2010-2017 / 2018-2020 / 2021-2024 boundaries. Added the leakage assertion. | Initially proposed random KFold. **Rejected**  -  would cause temporal leakage. | 2026-02-16 |
| 5 | **ECE as primary objective** | Human | Switched all ensemble tuning from accuracy to ECE after realising the project's central hypothesis requires calibration-optimised models. | Initially optimised all models for accuracy. **Corrected** by switching Optuna objective. | 2026-02-17 |
| 6 | **Model selection** | Collaborative | Chose the 3-tier structure (baseline → ensemble → deep learning) to span calibration properties. Selected Logistic Regression for its inherent calibration via sigmoid. | Suggested including LightGBM alongside XGBoost. Generated the Optuna tuning loops. | 2026-02-17 |
| 7 | **Portfolio simulation** | Collaborative | Designed the quintile-based equal-weight portfolio. Chose half-Kelly threshold for stock selection. Specified the 20% cutoff following Jegadeesh & Titman. | Generated the backtest loop. I verified monthly returns against manual spot checks. | 2026-02-18 |
| 8 | **H1 statistical testing** | Human | Added paired t-test, bootstrap CI, Cohen's d, Shapiro-Wilk, and Bonferroni correction. The agent initially provided no formal significance test. | Generated the bootstrap resampling code after specification. | 2026-02-19 |
| 9 | **SHAP implementation** | Collaborative | Identified that default TreeExplainer failed on RF's array format. Added interventional perturbation with background data. Added permutation importance as a cross-check. | Default TreeExplainer code. **Failed** on RF arrays. I debugged and fixed. | 2026-02-20 |
| 10 | **Ablation study** | Human | Designed the pillar-removal ablation to test H2. Chose to retrain with identical hyperparameters for fair comparison. | Generated the ablation loop code. | 2026-02-20 |
| 11 | **Convergence analysis** | Collaborative | Required training curves for XGBoost ECE across boosting rounds and MLP loss across epochs. | Generated the callback-based logging. I verified the early-stopping pattern was visible. | 2026-02-21 |
| 12 | **Monte Carlo simulation** | Human | Specified 1,000 bootstrap resamples of monthly returns with iid caveat. | Generated the simulation and fan chart plotting code. | 2026-02-21 |
| 13 | **yfinance download fix** | Human | Detected that the agent's single-batch download caused 15% silent failures. Restructured into batches of 50 with per-batch validation and retry logic. | Single-batch download. **Caused data loss**  -  15% of tickers returned empty DataFrames silently. | 2026-02-22 |
| 14 | **Notebook structure** | Human | Split monolithic pipeline into 81 cells with markdown documentation. Organised into logical sections matching the report structure. | Initial monolithic script with no markdown. | 2026-02-23 |
| 15 | **DV-02 range check** | Collaborative | Identified that the automated range check was too strict ([-1, 1]) for monthly returns. Relaxed to (-0.95, 5.0) to accommodate legitimate extreme returns. | Implemented the check with [-1, 1] bounds. **Too strict**  -  flagged legitimate returns as errors. | 2026-02-24 |
| 16 | **VIF output** | Human | Identified that the notebook printed "All VIFs < 5.0" immediately after displaying VIFs of 2,809. Noted as a known output inconsistency in the agent log. | Generated the print statement without checking the actual values. **Misleading output.** | 2026-02-24 |
| 17 | **Report writing** | Human | Wrote all prose, interpreted all results, drew all conclusions. Structured as a tight story about decisions and evidence. | Provided initial section outlines and equation formatting. All narrative content is my own. | 2026-02-25 |
| 18 | **Test-set evaluation** | Human | Required test-set reliability diagrams to assess regime transfer. The agent initially only produced validation figures. | Generated the plotting code after specification. | 2026-02-26 |
| 19 | **Classification report** | Human | Added precision/recall/F1 reporting after model evaluation. Missing from the initial pipeline. | Omitted classification metrics entirely. **Added** after I identified the gap. | 2026-02-26 |
| 20 | **Figure cropping** | Collaborative | Identified that all figures had burned-in "Figure X:" titles that duplicated the report captions. Cropped all 20 images. | Generated the cropping script. I verified each image visually. | 2026-03-08 |

---

## Section 2: Verification Methods  -  How I Tested the AI's Code

| # | Code Block | Verification Method | Result | Action Taken |
|---|------------|---------------------|--------|--------------|
| 1 | **yfinance batch download** | Counted non-null tickers after download. Expected 457; got 388 with single-batch. | 15% data loss confirmed. | Restructured to batches of 50 with per-batch row-count validation. |
| 2 | **Temporal split assertion** | `assert df_train['date'].max() < df_val['date'].min()` for all split boundaries. | Assertion passed for all boundaries. | Accepted. |
| 3 | **Feature leakage check** | Verified that all features use `shift(1)` or t-1 window endings. Checked that FRED variables are lagged 1 month. | No look-ahead detected in features. FRED revised-vintage risk noted as limitation. | Accepted with caveat in report. |
| 4 | **Optuna ECE objective** | Printed the objective function source. Confirmed `calculate_ece(y_val, y_prob)` not `accuracy_score`. | ECE confirmed as objective for ensembles. LR uses default (inherently calibrated). | Accepted. |
| 5 | **Elo rating dynamics** | Plotted 5 stock trajectories over 2010-2024. Verified ratings start at 1500 and diverge based on outperformance. Checked that K=20 produces ~240 points for 12 consecutive wins. | Trajectories sensible. AAPL rises, underperformers fall. K=20 dynamics stable. | Accepted. Figure A9 in appendix. |
| 6 | **Bayesian shrinkage** | Scatter plotted raw vs shrunk returns. Verified extreme values are pulled toward sector mean. Checked shrinkage weight = 0.181. | Shrinkage behaves correctly. Extreme stocks regularised most. | Accepted. Figure A10 in appendix. |
| 7 | **ECE calculation** | Manually computed ECE for a toy example (10 predictions, 2 bins). Compared against function output. | Values match within floating-point tolerance. | Accepted. |
| 8 | **Brier decomposition** | Verified that REL + UNC - RES = Brier score for Random Forest. Checked component values against Murphy (1973) definitions. | Decomposition sums correctly. REL component matches ECE trend. | Accepted. |
| 9 | **Portfolio returns** | Spot-checked 3 random months by manually computing equal-weight returns from adjusted close prices. | Returns match within rounding tolerance (< 1 bps). | Accepted. |
| 10 | **Paired t-test** | Verified t-statistic by computing manually: mean(diff) / (std(diff) / sqrt(48)). | Manual calculation matches scipy output exactly. | Accepted. |
| 11 | **Bootstrap CI** | Checked that 5,000 resamples produce a distribution centred near the point estimate. Verified percentile method for CI construction. | CI [-0.080, 0.204] is centred near 0.062 (point estimate). Distribution approximately normal. | Accepted. |
| 12 | **SHAP interventional** | Compared SHAP importance rankings against permutation importance. | Broad agreement between methods. bb_position and vix_change_1m rank consistently high. | Accepted. Cross-check in Figure A6. |
| 13 | **Ablation ECE values** | Re-ran ablation twice to check stability. | ECE values varied by ±0.001 across runs, consistent with fold-level SE of ~0.010. | Accepted with noise-band caveat in report. |
| 14 | **Confusion matrix** | Verified that TP + FP + TN + FN = N for validation set. Checked recall = 0.018 for underperform class. | All values consistent. RF predicts "Outperform" for 98% of stocks. | Accepted. Figure A8 in appendix. |
| 15 | **DeLong test** | Verified XGBoost AUC = 0.529 against manual `roc_auc_score()` computation. | Values match. p = 0.12 confirms AUC not distinguishable from 0.50. | Accepted. |
| 16 | **Monte Carlo** | Checked that 100% of 1,000 paths end positive. Verified median terminal return = 63.1%. | Results reflect the strong 2021-2024 market, not model skill. iid caveat noted. | Accepted with caveat. |
| 17 | **Learning curve** | Verified that both train and validation accuracy plateau near 50% as training set grows. | Noise ceiling confirmed. Model is data-saturated, not capacity-limited. | Accepted. Figure A5 in appendix. |
| 18 | **XGBoost convergence** | Plotted ECE at each boosting round. Verified minimum at round 10 and degradation to 0.093 by round 150. | Early stopping is essential for calibration. Production model uses best-trial hyperparameters. | Accepted. Figure 6 in report. |
| 19 | **VIF computation** | Manually verified VIF for elo_prob by regressing it on all other features. | VIF = 2,809 confirmed. The notebook's "All VIFs < 5" print statement is incorrect. | Noted as known output inconsistency. Report discusses honestly. |

---

## Section 3: Course Corrections  -  Where the AI Erred and How I Fixed It

| # | What Went Wrong | Root Cause | How I Fixed It | Impact |
|---|----------------|------------|----------------|--------|
| 1 | **Single-batch yfinance download** lost 15% of tickers silently. | The agent called `yf.download()` with all 503 tickers at once. Yahoo Finance silently drops tickers that timeout. | Restructured into batches of 50 with per-batch row-count validation and 3 retries for failed batches. | Recovered 69 tickers (457 vs 388). |
| 2 | **Random KFold for hyperparameter tuning** would have caused temporal leakage. | The agent used `sklearn.model_selection.KFold` with `shuffle=True`, mixing future data into training folds. | Replaced with expanding-window temporal CV (3 folds: train 2010-13/val 14-15, train 2010-15/val 16-17, train 2010-17/val 18). | Eliminated temporal leakage in Optuna tuning. |
| 3 | **All models optimised for accuracy** instead of ECE. | The agent defaulted to `accuracy_score` as the Optuna objective for all models. | Switched ensemble objectives to `calculate_ece()`. Left LR with default (inherently calibrated). | Aligned tuning with the project's central hypothesis. |
| 4 | **SHAP TreeExplainer failed** on Random Forest's multi-output array format. | RF in sklearn returns arrays with shape (n_samples, n_classes), which TreeExplainer does not handle by default. | Added `feature_perturbation='interventional'` with explicit background data (100 training samples). Added permutation importance as fallback. | Produced valid SHAP values for the calibration-best model. |
| 5 | **yfinance column format changed** between runs, causing empty DataFrames on `pd.concat()`. | The agent assumed a fixed MultiIndex column ordering. yfinance occasionally returns flat columns depending on the number of tickers. | Rewrote the download cell to detect column format (flat vs MultiIndex) and normalise before concatenation. | Eliminated intermittent download failures. |
| 6 | **DV-02 range check too strict** ([-1, 1] bounds). | Monthly returns can legitimately exceed ±100% for stocks that double or halve. | Relaxed to (-0.95, 5.0). A stock losing 95% or gaining 400% in a month is rare but legitimate. | Eliminated false-positive quality check failures. |
| 7 | **VIF print statement said "All VIFs < 5.0"** immediately after printing VIFs of 2,809. | The agent generated a hardcoded success message without checking the computed values. | Identified as a known output inconsistency. Report discusses VIFs honestly (Section 2) and tests Elo independence via ablation (Section 6.2). | Transparent handling of a cosmetic notebook error. |
| 8 | **No formal H1 significance test.** | The agent compared Sharpe ratios without any statistical test. | Added paired t-test, bootstrap CI (5,000 resamples), Cohen's d, Shapiro-Wilk normality test, and Bonferroni correction. | H1 result is now honestly reported as not statistically significant (p = 0.294). |
| 9 | **No test-set evaluation figures.** | The agent only produced validation-set reliability diagrams. | Required and generated test-set reliability diagrams (Figure A1) to assess regime transfer. | Revealed the 41% ECE degradation that is the most important finding. |
| 10 | **No classification report.** | The agent omitted precision/recall/F1 metrics from the pipeline. | Added `classification_report()` after each model evaluation. | Exposed RF's 98% "Outperform" prediction rate (recall = 0.018 for underperformers). |
| 11 | **Monolithic notebook structure.** | Initial code was a single continuous script with no markdown cells. | Split into 81 cells (43 markdown, 38 code) with section headers matching the report structure. | Notebook now serves as a research log per the brief. |
| 12 | **No convergence analysis.** | The agent trained models without logging per-round/epoch metrics. | Added callback-based ECE logging for XGBoost and loss logging for MLP. | Discovered XGBoost ECE degrades 4× without early stopping (Figure 6). |
| 13 | **Burned-in figure titles.** | The agent's matplotlib code included `plt.suptitle("Figure X: ...")` which duplicated report captions. | Cropped all 20 images to remove the burned-in titles. Report captions now serve as the sole figure labels. | Clean, professional figure presentation. |
| 14 | **No power analysis.** | The agent did not assess whether the test window was long enough to detect the hypothesised effect. | Added post-hoc power analysis: 0.72 power for d = 0.32, leaving a 28% miss probability. | Honest acknowledgement of limited statistical power. |
| 15 | **Missing Brier reference.** | The agent cited "Brier score" in table captions without adding Brier (1950) to the reference list. | Added: Brier, G.W. (1950) 'Verification of forecasts expressed in terms of probability', Monthly Weather Review, 78(1), 1-3. | Complete reference list. |

---

## Summary

The agent was a productive collaborator for code generation, boilerplate, and visualisation. It failed systematically on decisions requiring financial domain knowledge  -  particularly temporal discipline (items 2, 3), data quality monitoring (items 1, 5, 6), and statistical rigour (item 8). Every correction required human judgement about what constitutes valid evidence in a financial prediction context. In hindsight, front-loading the temporal CV and leakage checks before any training code would have saved the most debugging time.
