import numpy as np
import pandas as pd
import os

import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Load data
print("Loading data")
train_df=pd.read_csv("./data/feature/train.csv")
print("Data loaded successfully.")

# Split data into features and target
print("Splitting data into features and target...")
X = train_df.drop(columns=['driver_acceptance'])
y = train_df['driver_acceptance']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Data splitting complete.")

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
print("Feature standardization complete.")

print("Saving the scaler...")
scaler_path = os.path.join("models", "scaler.pkl")
if not os.path.exists("models"):
    os.makedirs("models")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved successfully.")

# Train the model
print("Training the model...")
model = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=1,
    class_weight='balanced',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)   # suppress logs
    ]
)
print("Model training complete.")

# Save the model
print("Saving the model...")
model_path = os.path.join("models", "lightgbm_model.pkl")
if not os.path.exists("models"):
    os.makedirs("models")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")
