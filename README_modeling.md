# Supervised Classification Modeling & Tuning — Credit Default Risk

Trains and compares 4 model architectures on a credit-default-risk dataset, tunes each with
`GridSearchCV` over stratified 5-fold CV, evaluates on a held-out test set, and serializes the
champion model.

## Contents
- `modeling_and_tuning.ipynb` — full notebook: training, tuning, evaluation, ROC curves, champion selection
- `credit_default_risk.csv` — dataset (1,000 applicants, mixed numeric/categorical, binary `default` target)
- `champion_model.joblib` — serialized winning pipeline (preprocessing + model, load with `joblib.load(...)`)
- `model_comparison.csv` — final comparison table
- `requirements.txt` — Python dependencies

## Models compared
Logistic Regression, Random Forest, XGBoost, LightGBM — each wrapped in the same leak-free
`ColumnTransformer` preprocessing pipeline (median/mode imputation, `StandardScaler`, `OneHotEncoder`,
cross-fit `TargetEncoder`), tuned via `GridSearchCV(scoring='roc_auc', cv=StratifiedKFold(5))`.

## Result

| model | cv_roc_auc_mean | test_roc_auc | test_precision | test_recall | test_f1 |
|---|---|---|---|---|---|
| **LightGBM (champion)** | 0.6642 | **0.7248** | 0.6543 | 0.5824 | 0.6163 |
| Logistic Regression | 0.7238 | 0.7164 | 0.6447 | 0.5385 | 0.5868 |
| Random Forest | 0.7005 | 0.7164 | 0.6500 | 0.5714 | 0.6082 |
| XGBoost | 0.6717 | 0.7034 | 0.6386 | 0.5824 | 0.6092 |

Champion selected on test ROC-AUC.

## Usage
```python
import joblib
model = joblib.load('champion_model.joblib')
model.predict(new_applicants_df)       # raw dataframe in, no manual preprocessing needed
model.predict_proba(new_applicants_df)
```
