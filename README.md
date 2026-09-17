# Credit Default Risk — Real-Time ML Inference API

A production-style FastAPI microservice serving the champion LightGBM credit-risk model, containerized with Docker and covered by a unit test suite. This is the final stage of an end-to-end ML pipeline: problem framing → feature engineering → model training/tuning → real-time serving.

**Live locally at:** `http://localhost:8000` · **Interactive docs:** `http://localhost:8000/docs`

---

## 1. System architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  Raw applicant data  │ --> │  Preprocessing        │ --> │  Model training        │
│  (credit_default_    │     │  ColumnTransformer:    │     │  & tuning:             │
│  risk.csv)            │     │  - median/mode impute  │     │  LR, RF, XGBoost,      │
│                        │     │  - StandardScaler      │     │  LightGBM              │
│                        │     │  - OneHotEncoder       │     │  GridSearchCV +        │
│                        │     │  - TargetEncoder       │     │  StratifiedKFold(5)    │
└─────────────────────┘     └──────────────────────┘     └───────────┬───────────┘
                                                                        │
                                                          champion selected on
                                                          test ROC-AUC (LightGBM)
                                                                        │
                                                                        v
                                                          ┌───────────────────────┐
                                                          │  champion_model        │
                                                          │  .joblib (single       │
                                                          │  serialized Pipeline:  │
                                                          │  preprocessing+model)  │
                                                          └───────────┬───────────┘
                                                                        │
                                                                        v
┌───────────────────────────────────────────────────────────────────────────────┐
│  FastAPI service (this repo)                                                   │
│  ┌───────────────┐   ┌────────────────┐   ┌──────────────────────────────┐    │
│  │ Pydantic       │-->│ request_to_     │-->│ pipeline.predict_proba(df)   │    │
│  │ schema         │   │ dataframe()     │   │ (imputation/encoding/model   │    │
│  │ validation     │   │                 │   │  all run inside the loaded   │    │
│  │ (422 on bad     │   │                 │   │  sklearn Pipeline — no logic │    │
│  │ input)          │   │                 │   │  is duplicated here)         │    │
│  └───────────────┘   └────────────────┘   └──────────────────────────────┘    │
│                     GET /health   POST /predict   POST /predict/batch          │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        v
                              Docker container (python:3.11-slim,
                              non-root user, healthcheck, pinned deps)
```

**Key design decision:** the entire preprocessing pipeline (imputers, scaler, one-hot encoder, target encoder) is serialized *inside* the same `sklearn.Pipeline` object as the model. The API never re-implements feature engineering — it builds a raw-feature `DataFrame` from the validated request and calls `.predict_proba()` once. This means the exact transformations used in training are guaranteed to run in production, with no train/serve skew.

## 2. Project structure

```
.
├── app/
│   ├── main.py              FastAPI app: /health, /predict, /predict/batch
│   ├── schemas.py           Pydantic request/response models + validation rules
│   └── model/
│       └── champion_model.joblib   serialized preprocessing+model Pipeline
├── tests/
│   └── test_api.py          30 unit tests: schema validation, response codes, happy path
├── Dockerfile
├── .dockerignore
├── requirements.txt         pinned runtime dependencies
├── requirements-dev.txt     + pytest/httpx for running tests
└── README.md
```

## 3. Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

## 4. Running with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

The image runs as a non-root user (`apiuser`), installs only `requirements.txt` (not the dev/test deps), and exposes a container `HEALTHCHECK` that polls `/health`.

## 5. API reference

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_version": "lightgbm-v1-2026-09-17"}
```

### `POST /predict`
Request:
```json
{
  "age": 34,
  "income_annual": 78000.0,
  "employment_type": "Salaried",
  "home_ownership": "Mortgage",
  "loan_purpose": "Auto",
  "education": "Bachelor",
  "loan_amount": 15000.0,
  "credit_score": 690.0,
  "debt_to_income_ratio": 0.32
}
```
Response (`200`):
```json
{
  "default_probability": 0.4014,
  "predicted_class": 0,
  "risk_band": "medium",
  "model_version": "lightgbm-v1-2026-09-17"
}
```

`income_annual`, `education`, and `credit_score` are optional — the underlying pipeline's imputers handle missing values, matching real-world applicant data. All other fields are required; a missing required field or an invalid value (bad enum, out-of-range number, wrong type) returns `422` with a field-level error list.

### `POST /predict/batch`
Same schema, wrapped as `{"applicants": [ {...}, {...} ]}` (1–500 items), returns `{"results": [ {...}, ... ]}`.

## 6. Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

30 tests covering: health check, valid-request happy path, response schema/value bounds, optional-field handling, required-field omission (`422`), invalid enum/range/type values (`422`), batch inference, and docs availability. Verified passing in this repo's build environment before packaging.

## 7. Model details

| model | cv_roc_auc_mean | test_roc_auc | test_precision | test_recall | test_f1 |
|---|---|---|---|---|---|
| **LightGBM (champion)** | 0.6642 | **0.7248** | 0.6543 | 0.5824 | 0.6163 |
| Logistic Regression | 0.7238 | 0.7164 | 0.6447 | 0.5385 | 0.5868 |
| Random Forest | 0.7005 | 0.7164 | 0.6500 | 0.5714 | 0.6082 |
| XGBoost | 0.6717 | 0.7034 | 0.6386 | 0.5824 | 0.6092 |

Full training/tuning notebook, comparison logic, and the leak-free `ColumnTransformer` build are in the modeling-stage notebook of this pipeline (`modeling_and_tuning.ipynb`), not duplicated here to keep this repo focused on serving.

## 8. Known limitations

- **Synthetic training data.** `credit_default_risk.csv` (1,000 rows) was generated to exercise the full pipeline end-to-end, not sourced from a real lender. Treat the ROC-AUC numbers above as a pipeline sanity check, not a production benchmark — retrain on real, larger, longitudinal data before using this for actual lending decisions.
- **No auth/rate-limiting.** This service has no authentication, request throttling, or audit logging — add these before exposing it outside a trusted network.
- **Single model version served.** There's no A/B or shadow-deployment mechanism; swapping models means rebuilding the image with a new `champion_model.joblib`.
- **Docker build untested in this environment** (no Docker daemon available where this was built) — the image was validated by resolving `requirements.txt` in an isolated virtualenv and confirming the app imports and serves correctly outside Docker; verify the actual `docker build`/`docker run` on your machine before treating it as certified.
