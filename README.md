# Predictive Maintenance for IoT Devices

## Overview
This repository contains an LSTM-based forecasting model for predictive maintenance in manufacturing settings. It analyzes time series data from IoT sensors using Python and Keras.

## Notebooks
These notebook includes:
- **Data Import**: Load and preprocess NASA C-MAPSS dataset.
- **Exploratory Data Analysis (EDA)**: Analyze and visualize data.
- **Time Series Analysis**: Conduct stationarity tests and Fourier Transform.
- **Feature Engineering**: Create time-series features.
- **Model Training**: Train and evaluate models (Linear Regression, Random Forest, XGBoost, Catboost, LSTM, GRU).
- **Hyperparameter Tuning**: Optimize XGBoost model using GridSearchCV
- **Model Evaluation**
  Evaluation Metric: Root Mean Squared Error (RMSE)
  Results:
  Linear Regression RMSE: 1.707
  Random Forest RMSE: 1.517
  XGBoost RMSE: 1.461
  CatBoost RMSE : 1.15
  LSTM RMSE: 2.731
  GRU RMSE: 2.294
- Explored Hybrid Models (Catboost + Lstm), Model Ensembles of top 3 models as well but CatBoost alone still remained as best performer.

**MultiVariate Time Series for Predictive Maintenance - All Datasets**
This notebook includes:
- **Data Impor**t: Load multiple datasets from NASA C-MAPSS.
- **Data Processing**: Combine and preprocess the datasets.
- **Feature Engineering**: Create time-series features.
- **Model Training**:
  1. Train and evaluate models (Linear Regression, Random Forest, XGBoost, CatBoost).
  2. Model Stacking: Implement stacking of multiple models.
  3. Deep Learning Models: Train LSTM and GRU models.
- **Model Evaluation**:
  Evaluation Metric: Root Mean Squared Error (RMSE)
  Results:
    CatBoost: 0.747
    XGBoost: 0.975
    Random Forest: 1.285
    Linear Regression: 1.627
- CatBoost remained as best performer.
## Dataset
The dataset used is the NASA C-MAPSS dataset, available at the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

## Results
The notebooks demonstrate the development and evaluation of predictive models for estimating the Remaining Useful Life (RUL) of equipment, compared based on RMSE values. 
