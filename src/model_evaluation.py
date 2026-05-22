import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import json

import pickle

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

class ClassificationEvaluator:
    
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        
        self.y_prob = model.predict_proba(X_val)[:, 1]
        self.y_pred = model.predict(X_val)
    
    def basic_metrics(self):
        print("Accuracy :", accuracy_score(self.y_val, self.y_pred))
        print("Precision:", precision_score(self.y_val, self.y_pred))
        print("Recall   :", recall_score(self.y_val, self.y_pred))
        print("F1 Score :", f1_score(self.y_val, self.y_pred))
        print("ROC-AUC  :", roc_auc_score(self.y_val, self.y_prob))
    
    def classification_report(self):
        print("\nClassification Report:\n")
        print(classification_report(self.y_val, self.y_pred))
    
    def confusion_matrix_plot(self):
        cm = confusion_matrix(self.y_val, self.y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    def roc_curve_plot(self):
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(self.y_val, self.y_prob)
        auc = roc_auc_score(self.y_val, self.y_prob)
        
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
        
        return auc
    
    def threshold_analysis(self):
        print("\nThreshold Analysis:")
        for t in [0.4, 0.5, 0.6]:
            y_pred_t = (self.y_prob >= t).astype(int)
            cm = confusion_matrix(self.y_val, y_pred_t)
            print(f"\nThreshold = {t}")
            print(cm)
            
# Load model and scaler
print("Loading model and scaler...")
model_path = "models/lightgbm_model.pkl"
scaler_path = "models/scaler.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print("Model and scaler loaded successfully.")

#Load validation data
print("Loading validation data...")
val_df = pd.read_csv("./data/feature/test.csv")
print("Validation data loaded successfully.")

# Prepare validation data
print("Preparing validation data...")
X_val = val_df.drop(columns=['driver_acceptance'])
y_val = val_df['driver_acceptance']
X_val = scaler.transform(X_val)
print("Validation data preparation complete.")

evaluator = ClassificationEvaluator(model, X_val, y_val)

evaluator.basic_metrics()
evaluator.classification_report()
evaluator.confusion_matrix_plot()
evaluator.threshold_analysis()

output_path = os.path.join("reports", "evaluation_report.json")
report = {
    "accuracy": accuracy_score(y_val, evaluator.y_pred),
    "precision": precision_score(y_val, evaluator.y_pred),
    "recall": recall_score(y_val, evaluator.y_pred),
    "f1_score": f1_score(y_val, evaluator.y_pred),
    "roc_auc": roc_auc_score(y_val, evaluator.y_prob)
} 
print(f"\nSaving evaluation report to {output_path}...")
if not os.path.exists("reports"):
    os.makedirs("reports")
with open(output_path, "w") as f:
    json.dump(report, f, indent=4)
print("Evaluation report saved successfully.")
