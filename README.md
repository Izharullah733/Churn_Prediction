House Price Prediction Project
Welcome to the House Price Prediction Project! This repository contains a machine learning project that uses linear regression to predict house prices based on the California Housing dataset. The project includes data exploration, preprocessing, model training, evaluation, and hypothesis testing.

Project Overview
Dataset: The project uses the housing.csv dataset(California house data from kaggle), which includes features like housing median age, total rooms, population, median income, etc., to predict median house values.
Model: A linear regression model is built after log-transforming the data for better distribution and handling outliers.

Key Steps:
Data loading and cleaning (handling nulls, encoding categorical variables).
Exploratory Data Analysis (EDA): Descriptive stats, correlation matrix, histograms, boxplots, scatter plots.
Data transformation (log transformation).
Model training and evaluation (MSE, R² score, actual vs predicted plots).
Hypothesis testing on the overall model and individual features (F-test and t-tests).


Files:
notebooks/main.ipynb: Main notebook for data exploration, preprocessing, model building, and evaluation.
notebooks/hypothesis_testing.ipynb: Notebook focused on detailed hypothesis testing for the model and features.
data/housing.csv: The dataset file (add it here if it's not too large; otherwise, provide a download link in this README).



The model achieves around 70% accuracy (R² score) and MSE 0.11 after preprocessing.

Prerequisites
Python 3.8+
Jupyter Notebook or JupyterLab for running the .ipynb files.