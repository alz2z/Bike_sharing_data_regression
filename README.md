# Seoul Bike Sharing Demand Prediction

A time series regression project that predicts hourly bike rental demand in Seoul using machine learning. The project compares multiple models across two feature engineering strategies, with full hyperparameter tuning and model interpretability analysis.

## Overview

Seoul operates a large public bike-sharing system with demand that varies by hour, season, and weather conditions. This project builds and evaluates regression models to forecast the number of bikes rented per hour, which can inform resource allocation and fleet management decisions.

## Dataset

**SeoulBikeData.csv** — 8,760 hourly observations (one full year)

| Feature | Type | Description |
|---|---|---|
| Rented Bike Count | Target | Number of bikes rented per hour (0–3,556) |
| Hour | Temporal | Hour of the day |
| Temperature | Weather | °C |
| Humidity | Weather | % |
| Wind Speed | Weather | m/s |
| Visibility | Weather | 10m |
| Dew Point Temperature | Weather | °C |
| Solar Radiation | Weather | MJ/m² |
| Rainfall | Weather | mm |
| Snowfall | Weather | cm |
| Seasons | Categorical | Winter, Spring, Summer, Autumn |
| Holiday | Categorical | Holiday / No Holiday |
| Functioning Day | Categorical | Yes / No |

## Feature Engineering Strategies

Two strategies were tested to incorporate temporal dependencies:

- **Long AR** — Autoregressive lags at 24h, 48h, 72h, and 168h (1 week), capturing weekly seasonality (~30 features)
- **VAR(1–3)** — Short-term lags at 1, 2, and 3 hours for both the target and weather variables, capturing immediate dependencies (~45 features)

## Models

Four regression models were trained for each feature strategy (8 total pipelines):

| Model | Notes |
|---|---|
| Ridge Regression | L2 regularization baseline |
| Random Forest | 200–400 trees |
| Gradient Boosting | Sequential boosting, shallow trees |
| XGBoost | Advanced gradient boosting |

**Validation**: TimeSeriesSplit (5 folds) with GridSearchCV for hyperparameter tuning. An 80/20 time-based train/test split is used to prevent data leakage.

**Metrics**: RMSE (primary), MAE, R²

**Baselines**: mean predictor, naive lag-168 predictor

## Model Interpretability

- Permutation feature importance (model-agnostic)
- Tree-based feature importance
- SHAP values for local explanations

Key drivers of demand: hour of day, temperature, humidity, solar radiation, seasonal indicators, and short-term lag features.

## Project Structure

```
├── 1030_data_project_final.ipynb   # Main notebook
├── 1030_data_project_final.html    # HTML export
├── 1030_data_project_final.pdf     # PDF report
├── SeoulBikeData.csv               # Source data
├── saved_artifacts/                # Trained model pipelines (.pkl) and feature names
└── figures/                        # Plots and visualizations
```

## Tech Stack

- **Data**: pandas, numpy
- **Modeling**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn
- **Interpretability**: SHAP
- **Notebook**: Jupyter
