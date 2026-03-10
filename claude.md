# Rules I Gave the Agent Before Starting

This document records the instructions I wrote for Claude (Anthropic) before beginning the coding phase of the project. The idea was to prevent the most common AI mistakes in financial ML -- particularly temporal leakage, which the agent repeatedly introduced in early iterations before I defined these rules explicitly.

In hindsight, I should have written this file *first*, before generating any code. Most of the corrections documented in `agent_collaboration_log.md` (especially D2, D3, and D8) happened because the agent started coding before I had articulated the constraints.

---

## Temporal Rules (the ones that matter most)

These are non-negotiable. Every correction I had to make to the agent's code involved one of these:

1. **Never shuffle data across time.** All splits must follow strict calendar order. The agent's first attempt used `sklearn.KFold(shuffle=True)`, which mixed 2019 validation data into 2012 training folds. I caught this and replaced it with expanding-window temporal CV.

2. **Training ends December 2017.** Nothing from 2018 onwards can touch the training set. This sounds obvious, but the agent never questioned it -- it would have happily trained on the full dataset if I hadn't specified the split.

3. **The test set (2021-2024) is evaluated exactly once.** No iterative tuning, no peeking, no "let me just check one thing." The portfolio simulation runs once and the results are what they are.

4. **Every feature must use `shift(1)` or a trailing window ending at t-1.** The agent built features using same-period data on three separate occasions. Each time I had to manually verify the temporal alignment.

5. **FRED macro variables are lagged one month.** I also flag that FRED publishes *revised* data, not real-time vintages, which introduces a subtle look-ahead bias that I could not correct without a real-time data subscription.

---

## Modelling Rules

6. **The Optuna objective for all ensemble models is `calculate_ece()`, not `accuracy_score()`.** This is the project's central hypothesis. The agent initially set accuracy as the objective for all six models. I changed it for the ensembles and left it for the baselines (Dummy has no tuning; LR is inherently calibrated).

7. **Random Forest is the calibration-best model. XGBoost is the accuracy-best.** Both are tree ensembles, so comparing them isolates the bagging-vs-boosting effect on calibration.

8. **The MLP is included deliberately as a calibration failure case.** Deep networks are known to be poorly calibrated (Guo et al., 2017), and I wanted to see whether ECE-targeted tuning could rescue it. It could not.

---

## Portfolio Rules

9. **Top quintile (~91 stocks) by predicted probability each month.** I considered deciles but the signal is too weak to support finer sorting.

10. **Equal-weighted, not Kelly-proportional.** With AUC barely above 0.50, Kelly fractions would amplify estimation noise into extreme positions. Equal weighting is more robust.

11. **Transaction costs: 5 basis points per leg.** Applied at every monthly rebalance.

---

## Statistical Discipline

12. **H1 requires a paired t-test on the 48 monthly return differences**, plus bootstrap CI (5,000 resamples), Cohen's d, Shapiro-Wilk normality check, and Bonferroni correction for two hypotheses.

13. **Report honest results.** If the model does not work, say so. The agent's instinct was to present everything positively. I overrode this in every section where the results were ambiguous or negative.

---

## Known Issues I Chose Not to Fix

- **VIF output** prints "All VIFs < 5.0" despite max VIF being 2,809. This is a cosmetic bug in the notebook assertion cell. The report addresses the collinearity honestly.
- **FRED revised vintages** introduce look-ahead bias. I flag this as a limitation rather than correcting it, because real-time FRED data requires a paid subscription.
- **Survivorship bias** in the S&P 500 constituent list inflates apparent skill by 50-100 bps annually. I could not correct this without historical constituent data.

---

## Verification Checklist (what I checked before every commit)

- [ ] No temporal leakage: `max(train_dates) < min(val_dates) < min(test_dates)`
- [ ] ECE is the Optuna objective for all ensembles
- [ ] All features use `shift(1)` or trailing windows ending at t-1
- [ ] Test set has not been touched during tuning
- [ ] Figures save to `figures/` directory
- [ ] No hardcoded success messages that contradict actual outputs
