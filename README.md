# **Bank Customer Churn Prediction**

**Overview**
This project aims to predict customer churn in a banking dataset using machine learning and deep learning models. The goal is to identify customers likely to leave the bank and understand the key factors driving churn. The project includes data preprocessing, feature engineering (handling class imbalance with SMOTE), model comparison, and visualizations to analyze performance and churn drivers.
The dataset used is the Churn Modelling Dataset from Kaggle, which includes customer details like credit score, geography, gender, age, tenure, balance, and more. The project compares Logistic Regression, Naive Bayes, KNN, a simple ANN, and a tuned DNN, with the DNN achieving high accuracy, precision, and recall.

**Dataset**

**Source:** Kaggle - Churn Modelling Dataset
**Description:** Contains customer data such as CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and Exited (target variable indicating churn).
Note: Download Churn_Modelling.csv and place it in the project directory or update the file path in the script.

**Project Structure**

**bank_churn_prediction.py:** Main script for data preprocessing, feature engineering, model training, evaluation, and visualization.
**requirements.txt:** Lists required Python libraries.
**.gitignore:** Specifies files and directories to exclude from version control.
**README.md:** This file, providing project details and instructions.

**Prerequisites**

Python: Version 3.11.9 (developed using VS Code)
Libraries:
pandas
numpy
seaborn
matplotlib
scikit-learn
imblearn
tensorflow
optuna






**Usage**

**Data Preprocessing:**
Drops irrelevant columns (RowNumber, CustomerId, Surname).
Encodes categorical variables (Geography, Gender).
Standardizes features and applies SMOTE to handle class imbalance.


**Feature Engineering:**
Uses Logistic Regression coefficients to identify key churn drivers (e.g., Age, Balance, Geography).


**Models:**
**Machine Learning:** Logistic Regression, Naive Bayes, KNN.
**Deep Learning:** Simple ANN and tuned DNN (optimized with Optuna for high recall).


**Visualizations:**
Feature importance bar plot.
Confusion matrices for each model.
Bar plots comparing model performance (Accuracy, Precision, Recall, F1 Score, ROC AUC).




**Results**


The tuned DNN achieves the highest accuracy (85.4%), precision (62.6%), and F1 Score (63.2%), with a strong recall (63.9%), making it effective for identifying churned customers.
**Key Factors Driving Customer Churn**
Based on Logistic Regression coefficients, the top factors influencing churn are:

**Age:** 0.871169 (Older customers are more likely to churn)
**Balance:** 0.316488 (Higher balances correlate with churn)
**Geography:** 0.109871 (Certain regions show higher churn rates)
**EstimatedSalary:** 0.034020
**HasCrCard:** 0.018192

**Visualizations**

Feature Importance: Bar plot showing the impact of each feature on churn.
Confusion Matrices: Display true positives, false positives, etc., for each model.
Model Comparison: Bar plots for Accuracy, Precision, Recall, F1 Score, and ROC AUC.


**Future Improvements**

Test ensemble models (e.g., Random Forest, XGBoost) for potentially better performance.
Explore advanced feature engineering (e.g., interaction terms, polynomial features).
Implement cross-validation for more robust evaluation.
Optimize DNN further with additional layers or advanced architectures (e.g., LSTM).


This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, reach out via GitHub Issues or contact Izharullah733.
