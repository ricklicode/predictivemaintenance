#!/usr/bin/env python
"""
Script to train predictive maintenance models for each failure type (TWF, HDF, PWF, OSF, RNF).
Uses Random Forest models with engineered features and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import pickle

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
df['Speed_Torque_Ratio'] = df['Rotational speed [rpm]'] / (df['Torque [Nm]'] + 1e-6)  # Avoid division by zero
df['Temp_Ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']

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
    'Tool_Torque [minNm]',
    'Speed_Torque_Ratio',
    'Temp_Ratio'
]

X = df[features]

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

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
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Initialize GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model with hyperparameter tuning
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Get best model
    model = grid_search.best_estimator_
    print(f"\nBest parameters for {failure_mode}:")
    print(grid_search.best_params_)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nPerformance Metrics for {failure_mode}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
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
    
    # Permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance = pd.DataFrame({
        'Feature': features,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 5 Features by Permutation Importance for {failure_mode}:")
    for i, row in perm_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Save the model
    with open(f'failure_type_results/{failure_mode.lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title(f'Feature Importance for {failure_mode}')
    plt.tight_layout()
    plt.savefig(f'failure_type_results/{failure_mode.lower()}_feature_importance.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {failure_mode}')
    plt.legend(loc="lower right")
    plt.savefig(f'failure_type_results/{failure_mode.lower()}_roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {failure_mode}')
    plt.legend(loc="lower left")
    plt.savefig(f'failure_type_results/{failure_mode.lower()}_pr_curve.png')
    plt.close()
    
    # Save results
    results[failure_mode] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'best_params': grid_search.best_params_,
        'feature_importance': feature_importance.to_dict(),
        'permutation_importance': perm_importance.to_dict()
    }

# Save all results
with open('failure_type_results/model_results.txt', 'w') as f:
    for failure_mode, result in results.items():
        f.write(f"\n{failure_mode} Results:\n")
        f.write(f"Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {result['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {result['avg_precision']:.4f}\n")
        f.write(f"Best Parameters: {result['best_params']}\n")
        f.write("\nTop 5 Features by Importance:\n")
        for feature, importance in list(result['feature_importance']['Importance'].items())[:5]:
            f.write(f"  {result['feature_importance']['Feature'][feature]}: {importance:.4f}\n") 