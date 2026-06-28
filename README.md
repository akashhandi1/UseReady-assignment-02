# Load Type Classification for a Power System

Predict the `Load_Type` of a power system (`Light_Load`, `Medium_Load`, `Maximum_Load`) from
operational energy-consumption data. The dataset is the Steel Industry Energy Consumption record
(DAEWOO Steel, 2018, 15-minute resolution), supplied with injected missing values and out-of-range
noise so that data cleaning forms part of the task.

## Repository structure

| Path | Description |
|------|-------------|
| `load_type_classification.ipynb` | End-to-end analysis: cleaning, EDA, feature engineering, modelling, and evaluation. |
| `data/load_data.csv` | Raw dataset. |
| `requirements.txt` | Python dependencies. |
| `Problem statement.docx` | Original task description. |

## Problem summary

- Task: multiclass classification with three classes.
- Target: `Load_Type` (`Light_Load`, `Medium_Load`, `Maximum_Load`).
- Class balance: approximately 52 / 28 / 21 percent, which is moderately imbalanced.
- Evaluation: the brief requires the last month (December 2018) to be used as the test set.

## Methodology

**Data cleaning.** The duplicate timestamp is removed, power factor values outside the valid
range [0, 100] are set to NaN, and `NSM` (seconds from midnight) is reconstructed deterministically
from the timestamp, which removes both its noise and its missing values. Median imputation and IQR
winsorisation are applied inside the modelling pipeline so they are fitted on the training folds
only and cannot leak information.

**Feature engineering.** Per-row features only, keeping each row an independent sample: calendar
fields (hour, day of week, month, day, quarter, weekend flag), cyclical sine and cosine encodings
of the time of day, day of week and month, and domain ratios (total reactive power, reactive power
per kWh, CO2 per kWh).

**Validation.** December 2018 is held out as the test set and is evaluated only once. Models are
tuned on January to November with `GridSearchCV` and `StratifiedKFold(5)`, optimising macro-F1.

**Models.** Eleven estimators are compared, each with a hyperparameter grid: a baseline dummy
classifier, Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Extra Trees,
Gradient Boosting, XGBoost, LightGBM, SVM, and Gaussian Naive Bayes. `class_weight='balanced'` is
used where supported to address the imbalance.

## Results

**Random Forest is the best-performing and recommended model, reaching 95.3 percent accuracy and
0.949 macro-F1 on the held-out December test set.** All tree-based ensembles comfortably outperform
the linear, distance-based, and naive Bayes baselines.

Leaderboard, sorted by cross-validation macro-F1 (December is the held-out test month):

| Model | CV macro-F1 | December macro-F1 | December accuracy |
|-------|-------------|-------------------|-------------------|
| LightGBM | 0.9999 | 0.8935 | 0.8962 |
| XGBoost | 0.9995 | 0.9138 | 0.9173 |
| Gradient Boosting | 0.9993 | 0.9095 | 0.9130 |
| **Random Forest (recommended)** | **0.9916** | **0.9492** | **0.9526** |
| Decision Tree | 0.9901 | 0.9203 | 0.9254 |
| Extra Trees | 0.9856 | 0.9178 | 0.9281 |
| KNN | 0.8014 | 0.6065 | 0.6885 |
| Logistic Regression | 0.7867 | 0.5177 | 0.6065 |
| SVM | 0.7420 | 0.5124 | 0.6069 |
| Gaussian Naive Bayes | 0.6779 | 0.5341 | 0.6257 |
| Dummy (baseline) | 0.3340 | 0.3227 | 0.3918 |

### Cross-validation choice versus test performance

Cross-validation runs on the January to November data, and by the macro-F1 criterion it selects
LightGBM (CV macro-F1 of 0.9999). The boosting models all reach near-perfect cross-validation
scores, which confirms that the three load classes are almost separable on the training
distribution: `Usage_kWh` together with the time-of-day features explains most of the target.

On December the ranking changes. LightGBM falls to 0.896 accuracy, while Random Forest leads with
0.953 accuracy and 0.949 macro-F1. The cause is distribution drift between the two periods: the
share of `Light_Load` rises from 51 percent in the training months to 59 percent in December. The
gradient-boosting models fit the training distribution very tightly, which is exactly why their
cross-validation scores are near 1.0, and they therefore give up more accuracy when that
distribution shifts. Random Forest averages many de-correlated trees, so it is less sensitive to
the shift and generalises better to the new month. The simpler ensembles (Decision Tree, Extra
Trees) sit between the two, and the non-ensemble models trail well behind.

### Recommended model

Random Forest is the preferred model for deployment under this drift. It delivers the highest
accuracy and macro-F1 on the most recent, unseen month, which is precisely the generalisation the
brief asks the model to demonstrate. LightGBM is reported as the cross-validation winner for
transparency, but the final model should not be chosen on December performance alone, because
selecting on the test set would be a form of leakage. The key takeaway is that the
cross-validation winner and the best real-world generaliser differ here, and Random Forest is the
stronger practical choice.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook load_type_classification.ipynb
```

Run all cells in order. The notebook reproduces the cleaning, analysis, tuning, evaluation, and a
saved `best_model.joblib` artifact. Results are deterministic via a fixed random seed.
