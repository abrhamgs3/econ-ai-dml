# econ-ai-dml

Double machine learning methods lab: causal effect estimation with ML nuisance functions.

## What this is

A small, readable lab for estimating a causal effect when you have many controls and do not want to impose a rigid functional form. It uses double machine learning (DML): two machine learning models soak up the predictable variation in the outcome and in the treatment, and the treatment effect is estimated from what is left, with cross-fitting so the same data is not used to both fit the nuisance models and estimate the effect.

The estimator is EconML's `LinearDML`, with random forest nuisance models for both the outcome and the treatment, and 3-fold cross-fitting.

## What is in here

- `main_dml.py` — smoke test on a tiny synthetic dataset: the effect of a minimum wage change on employment, controlling for age and education. Prints the estimated average treatment effect (ATE).
- `app.py` — Streamlit app: upload a CSV, pick the outcome, treatment and control columns, run DML, and see the estimated ATE.
- `src/dml_model.py` — the DML estimator (random forests, 3-fold cross-fitting).
- `src/load_data.py` — loads CSV files from `data/raw`.
- `src/clean_data.py` — drops rows with missing values.
- `src/features.py` — feature construction step (currently passes data through unchanged).
- `src/evaluate.py` — root mean squared error for a fitted model.
- `configs/config.yaml` — configuration file.

## Quickstart

```bash
pip install -r requirements.txt
python main_dml.py
```

To run the app:

```bash
streamlit run app.py
```

## Example

The smoke test in `main_dml.py` estimates the effect of `min_wage_change` on `employment`, controlling for `age` and `education`, and prints the estimated ATE.

## Status

Early-stage methods lab. The pipeline runs end to end on the synthetic demo and on uploaded CSVs through the app. The feature construction step is still a placeholder, and the cleaning step is limited to dropping missing values.
