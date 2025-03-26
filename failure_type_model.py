#!/usr/bin/env python
"""
Script to train predictive maintenance models for each failure type (TWF, HDF, PWF, OSF, RNF).
Uses Random Forest models with engineered features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import os
import time

# Create results directory if it doesn't exist
os.makedirs('failure_type_results', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data_path = 'uc_pred_mait_ds.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")

# Check failure modes
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for mode in failure_modes:
    print(f"{mode} failures: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)")

# Feature preparation with dummy variables and engineered features
print("\nPreparing features...")

# Extract product type and convert to dummy variables
df['Type_L'] = (df['Product ID'].str[0] == 'L').astype(int)
df['Type_M'] = (df['Product ID'].str[0] == 'M').astype(int)
df['Type_H'] = (df['Product ID'].str[0] == 'H').astype(int)

# Create engineered features
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Tool_Torque [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']

# Define features
features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]',
    'Type_L',
    'Type_M',
    'Type_H',
    'Power [W]',
    'Temp_Diff [K]',
    'Tool_Torque [minNm]'
]

X = df[features]

# Train models for each failure mode
results = {}

for failure_mode in failure_modes:
    print(f"\n--- {failure_mode} Failure Prediction Model ---")
    
    # Define target
    y = df[failure_mode]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples, Positive cases: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test set: {X_test.shape[0]} samples, Positive cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Train Random Forest model
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report for {failure_mode}:")
    print(report)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {failure_mode}:")
    print(conf_matrix)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 5 Features for {failure_mode}:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
    # Store results for this failure mode
    results[failure_mode] = {
        'model': model,
        'accuracy': accuracy,
        'auc': roc_auc,
        'report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }

# Create a combined model that predicts any type of failure
print("\n--- Combined Machine Failure Prediction Model ---")

# Define target
y_combined = df['Machine failure']

# Split the data
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# Train Random Forest model
start_time = time.time()
combined_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
combined_model.fit(X_train_combined, y_train_combined)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} seconds")

# Predictions
y_pred_combined = combined_model.predict(X_test_combined)
y_pred_proba_combined = combined_model.predict_proba(X_test_combined)[:, 1]

# Accuracy
accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
print(f"Accuracy: {accuracy_combined:.4f}")

# Classification report
report_combined = classification_report(y_test_combined, y_pred_combined)
print("\nClassification Report for Combined Model:")
print(report_combined)

# Confusion matrix
conf_matrix_combined = confusion_matrix(y_test_combined, y_pred_combined)
print("\nConfusion Matrix for Combined Model:")
print(conf_matrix_combined)

# Calculate ROC and AUC
fpr_combined, tpr_combined, _ = roc_curve(y_test_combined, y_pred_proba_combined)
roc_auc_combined = auc(fpr_combined, tpr_combined)
print(f"AUC-ROC: {roc_auc_combined:.4f}")

# Feature importance
feature_importance_combined = pd.DataFrame({
    'Feature': features,
    'Importance': combined_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Features for Combined Model:")
for i, row in feature_importance_combined.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Store combined model results
results['Combined'] = {
    'model': combined_model,
    'accuracy': accuracy_combined,
    'auc': roc_auc_combined,
    'report': report_combined,
    'confusion_matrix': conf_matrix_combined,
    'feature_importance': feature_importance_combined
}

# Write results to file
with open('failure_type_results/failure_type_models.txt', 'w') as f:
    f.write("Predictive Maintenance Models for Different Failure Types\n")
    f.write("====================================================\n\n")
    f.write(f"Dataset: {data_path}\n")
    f.write(f"Total samples: {len(df)}\n\n")
    
    # Write failure mode statistics
    f.write("Failure mode statistics:\n")
    for mode in failure_modes:
        f.write(f"- {mode}: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)\n")
    f.write(f"- Machine failure (any type): {df['Machine failure'].sum()} ({df['Machine failure'].mean()*100:.2f}%)\n\n")
    
    # Write model results for each failure mode
    for mode in failure_modes + ['Combined']:
        result = results[mode]
        
        if mode == 'Combined':
            f.write(f"\n--- Combined Machine Failure Prediction Model ---\n")
        else:
            f.write(f"\n--- {mode} Failure Prediction Model ---\n")
        
        f.write(f"Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {result['auc']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        cm = result['confusion_matrix']
        f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
        f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n\n")
        
        f.write("Classification Report:\n")
        f.write(result['report'])
        f.write("\n")
        
        f.write("Top 5 Feature Importance:\n")
        for i, row in result['feature_importance'].head(5).iterrows():
            f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")

print("\nResults saved to 'failure_type_results/failure_type_models.txt'")
print("Failure type models training and evaluation complete!") 