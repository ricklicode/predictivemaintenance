#!/usr/bin/env python
"""
Quick script to analyze the UC Predictive Maintenance dataset and train a model using scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import os

# Create results directory if it doesn't exist
os.makedirs('quick_results', exist_ok=True)

# Load the dataset
print("Loading dataset...")
data_path = 'uc_pred_mait_ds.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Machine failures: {df['Machine failure'].sum()} out of {len(df)} samples ({df['Machine failure'].mean()*100:.2f}%)")

# Check failure modes
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for mode in failure_modes:
    print(f"{mode} failures: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)")

# Preprocess the data
print("\nPreprocessing data...")
# Extract product type
df['Product_Type'] = df['Product ID'].str[0]

# Create a copy for processing
df_processed = df.copy()

# One-hot encode the product type
product_type_dummies = pd.get_dummies(df_processed['Product_Type'], prefix='Product_Type')
df_processed = pd.concat([df_processed, product_type_dummies], axis=1)

# Drop non-numeric columns that aren't needed for prediction
drop_cols = ['UDI', 'Product ID', 'Product_Type']
df_processed = df_processed.drop(drop_cols, axis=1)

# Define features and target
y = df_processed['Machine failure']
X = df_processed.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot correlation matrix
print("Generating correlation matrix...")
plt.figure(figsize=(12, 10))
correlation_matrix = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                           'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('quick_results/correlation_matrix.png')

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train models
print("\nTraining Random Forest model...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time
print(f"Random Forest training time: {rf_train_time:.2f} seconds")

print("\nTraining Logistic Regression model...")
start_time = time.time()
lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_train_time = time.time() - start_time
print(f"Logistic Regression training time: {lr_train_time:.2f} seconds")

# Evaluate models
print("\nEvaluating models with cross-validation...")
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"Random Forest CV accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(f"Logistic Regression CV accuracy: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# Get test set predictions
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nTest Set Results:")
print(f"Random Forest accuracy: {rf_accuracy:.4f}")
print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")

# Generate classification reports
print("\nRandom Forest Classification Report:")
rf_report = classification_report(y_test, rf_pred)
print(rf_report)

print("\nLogistic Regression Classification Report:")
lr_report = classification_report(y_test, lr_pred)
print(lr_report)

# Save classification reports to file
with open('quick_results/classification_reports.txt', 'w') as f:
    f.write("Random Forest Classification Report:\n")
    f.write(rf_report)
    f.write("\n\nLogistic Regression Classification Report:\n")
    f.write(lr_report)

# Plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)

plot_confusion_matrix(y_test, rf_pred, "Random Forest Confusion Matrix", 'quick_results/rf_confusion_matrix.png')
plot_confusion_matrix(y_test, lr_pred, "Logistic Regression Confusion Matrix", 'quick_results/lr_confusion_matrix.png')

# Plot ROC curves
def plot_roc_curve(y_true, y_pred_proba, model_name, filename):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    
    return roc_auc

rf_auc = plot_roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1], "Random Forest", 'quick_results/rf_roc_curve.png')
lr_auc = plot_roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1], "Logistic Regression", 'quick_results/lr_roc_curve.png')

# Compare ROC curves
plt.figure(figsize=(8, 6))
# Plot Random Forest ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {rf_auc:.4f})')

# Plot Logistic Regression ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_lr, tpr_lr, lw=2, label=f'Logistic Regression (AUC = {lr_auc:.4f})')

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.savefig('quick_results/roc_curves_comparison.png')

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('quick_results/feature_importance.png')

# Save the feature importance to CSV
feature_importance.to_csv('quick_results/feature_importance.csv', index=False)

# Save a summary of results
with open('quick_results/summary.txt', 'w') as f:
    f.write("Predictive Maintenance Model Results\n")
    f.write("===================================\n\n")
    f.write(f"Dataset: {data_path}\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Machine failures: {df['Machine failure'].sum()} ({df['Machine failure'].mean()*100:.2f}%)\n\n")
    
    f.write("Failure modes:\n")
    for mode in failure_modes:
        f.write(f"- {mode}: {df[mode].sum()} ({df[mode].sum() / len(df) * 100:.2f}%)\n")
    
    f.write("\nModel Performance (Test Set):\n")
    f.write(f"- Random Forest Accuracy: {rf_accuracy:.4f}\n")
    f.write(f"- Random Forest AUC: {rf_auc:.4f}\n")
    f.write(f"- Logistic Regression Accuracy: {lr_accuracy:.4f}\n")
    f.write(f"- Logistic Regression AUC: {lr_auc:.4f}\n\n")
    
    f.write("Top 5 Most Important Features:\n")
    for i, row in feature_importance.head(5).iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")

print("\nResults saved to 'quick_results' directory")
print("Analysis complete!") 