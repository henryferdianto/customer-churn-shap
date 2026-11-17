# Customer Churn Prediction & Explainable AI (SHAP) Retention Simulator

![ci](https://github.com/henryferdianto/customer-churn-shap/actions/workflows/ci.yml/badge.svg)

> **XGBoost/CatBoost churn scoring with SHAP-driven retention ROI simulation**

## Overview

A supervised churn prediction system that identifies high-risk subscribers, explains drivers with SHAP, and simulates the ROI of targeted retention campaigns.

## Approach

1. **Preprocessing** with class-imbalance handling (SMOTE / class weights).
2. **Modeling** with gradient boosting (XGBoost / CatBoost) and cross-validation.
3. **Explainability** using SHAP values at global and individual level.
4. **Retention simulator** estimating CLV saved under a campaign scenario.

## Repository Structure

```
customer-churn-shap/
├── src/train.py           # training pipeline
├── src/explain.py         # SHAP explanations
├── src/retention.py       # campaign ROI simulator
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `XGBoost` · `CatBoost` · `scikit-learn` · `imbalanced-learn` · `SHAP`

## Getting Started

```bash
git clone https://github.com/henryferdianto/customer-churn-shap.git
cd customer-churn-shap
pip install -r requirements.txt
```

```bash
python src/train.py --input data/raw/churn.csv
python src/explain.py --model models/churn.joblib --input data/raw/churn.csv
```


<!-- completed -->

## Quickstart with Sample Data

The repository ships a synthetic data generator so the full pipeline can be run
without any private dataset:

```bash
make sample && python -m src.train
python -m src.explain
python -m src.run_demo
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Project Status

This repository contains the project documentation and a runnable reference
implementation. A synthetic data generator is included so every pipeline stage
can be executed end-to-end; replace the sample data with your own datasets in
`data/raw/` to reproduce real results.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
