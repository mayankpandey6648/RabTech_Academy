# Feature Engineering & Preprocessing Pipeline

A leak-free Scikit-Learn preprocessing pipeline for a mixed numerical/categorical tabular classification problem.

## What this project demonstrates

- Train/test splitting before preprocessing
- Missing-value imputation
- Numerical feature scaling
- Categorical one-hot encoding
- `ColumnTransformer` and reusable `Pipeline`
- Numerical correlation analysis
- Random Forest feature importance
- Mutual information feature ranking
- Final model evaluation on an untouched test set
- Explicit data-leakage checks

## Dataset

**Adult Census Income dataset** loaded through Scikit-Learn/OpenML.

The target is whether annual income is `>50K`.

The notebook downloads the dataset automatically, so the raw dataset does not need to be committed to GitHub.

## Project structure

```text
feature-engineering-pipeline/
├── feature_engineering_pipeline.ipynb
├── README.md
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook feature_engineering_pipeline.ipynb
```

Or open the `.ipynb` file directly in GitHub/Google Colab.

## Data leakage prevention

The most important design choice is:

```text
Raw dataset
     |
     v
Train/Test Split
     |
     +--------------------+
     |                    |
     v                    v
  X_train               X_test
     |
     v
Preprocessing Pipeline
(imputation + scaling +
 one-hot encoding)
     |
     v
Model Training
     |
     v
Final evaluation on X_test
```

All preprocessing parameters are learned through the training pipeline rather than from the complete dataset.

## Expected deliverable

The main deliverable is:

`feature_engineering_pipeline.ipynb`

It contains executable code, explanations, visualizations, feature rankings, and evaluation results.
