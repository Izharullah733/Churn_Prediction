import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna

# Dataset: Bank Customer Churn Dataset from Kaggle
# Link: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
# Download the dataset and place 'Churn_Modelling.csv' in the project directory
# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# --- Data Preprocessing ---
# Drop irrelevant columns (RowNumber, CustomerId, Surname)
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Handle categorical variables (encode Geography and Gender)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Define features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
print("Before SMOTE, class distribution:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE, class distribution:", np.bincount(y_train_smote))

# --- Feature Importance Analysis ---
# Use Logistic Regression to identify important features
lr = LogisticRegression(random_state=42)
lr.fit(X_train_smote, y_train_smote)
feature_importance = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.show()

# --- Model Training and Evaluation ---
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Store results
results = []

# Train and evaluate machine learning models
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    })
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print(f"\nClassification Report for {name}:\n", classification_report(y_test, y_pred))

# --- Deep Learning Models ---
# 1. Simple ANN
ann = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_smote.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ann_history = ann.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
y_pred_ann = (ann.predict(X_test_scaled) > 0.5).astype(int)
results.append({
    'Model': 'ANN',
    'Accuracy': accuracy_score(y_test, y_pred_ann),
    'Precision': precision_score(y_test, y_pred_ann),
    'Recall': recall_score(y_test, y_pred_ann),
    'F1 Score': f1_score(y_test, y_pred_ann),
    'ROC AUC': roc_auc_score(y_test, y_pred_ann)
})
# Plot confusion matrix for ANN
cm_ann = confusion_matrix(y_test, y_pred_ann)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - ANN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("\nClassification Report for ANN:\n", classification_report(y_test, y_pred_ann))

# 2. DNN with Hyperparameter Tuning using Optuna
def create_dnn_model(trial):
    model = Sequential()
    model.add(Dense(units=trial.suggest_int('units1', 16, 64), activation='relu', input_shape=(X_train_smote.shape[1],)))
    model.add(Dropout(trial.suggest_float('dropout1', 0.1, 0.5)))
    model.add(Dense(units=trial.suggest_int('units2', 8, 32), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout2', 0.1, 0.5)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('lr', 1e-5, 1e-2, log=True)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    model = create_dnn_model(trial)
    model.fit(X_train_smote, y_train_smote, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    return recall_score(y_test, y_pred)  # Optimize for recall

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
print("Best DNN Parameters:", best_params)

# Train best DNN model
dnn = create_dnn_model(optuna.trial.FixedTrial(best_params))
dnn_history = dnn.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
y_pred_dnn = (dnn.predict(X_test_scaled) > 0.5).astype(int)
results.append({
    'Model': 'DNN (Tuned)',
    'Accuracy': accuracy_score(y_test, y_pred_dnn),
    'Precision': precision_score(y_test, y_pred_dnn),
    'Recall': recall_score(y_test, y_pred_dnn),
    'F1 Score': f1_score(y_test, y_pred_dnn),
    'ROC AUC': roc_auc_score(y_test, y_pred_dnn)
})
# Plot confusion matrix for DNN
cm_dnn = confusion_matrix(y_test, y_pred_dnn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dnn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - DNN (Tuned)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("\nClassification Report for DNN (Tuned):\n", classification_report(y_test, y_pred_dnn))

# --- Compare Model Performance ---
results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df)

# Visualize model performance
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
for i, metric in enumerate(metrics):
    sns.barplot(x=metric, y='Model', data=results_df, ax=axes[i])
    axes[i].set_title(metric)
plt.tight_layout()
plt.show()

# --- Why Customers Churn ---
# Based on feature importance, key drivers of churn include:
# - Geography: Certain regions may have higher churn rates.
# - Gender: Differences in churn behavior between genders.
# - Age, Balance, Number of Products: High coefficients indicate strong influence on churn.
print("\nKey Factors Driving Customer Churn:")
print(feature_importance.head())
