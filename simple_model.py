#!/usr/bin/env python
"""
Simple script to create a predictive maintenance model using scikit-learn.
Uses Random Forest and avoids complex preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import os
import time
import pickle

def train_and_save_models():
    """Train and save models for failure prediction, to be used by the web app"""
    # Create results directory if it doesn't exist
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('failure_type_results', exist_ok=True)
    
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
    
    # Prepare features
    print("\nPreparing features...")
    
    # Extract product types
    df['Product_Type'] = df['Product ID'].str[0]
    
    # Create dummy variables for product type
    product_dummies = pd.get_dummies(df['Product_Type'], prefix='Product_Type')
    df = pd.concat([df, product_dummies], axis=1)
    
    # Engineer additional features
    df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / 9.5488
    df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Tool_Torque [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
    
    # Define features for model
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                'Torque [Nm]', 'Tool wear [min]', 'Product_Type_L', 'Product_Type_M', 
                'Product_Type_H', 'Power [W]', 'Temp_Diff [K]', 'Tool_Torque [minNm]']
    
    X = df[features]
    
    # Train a model for each failure type
    models = {}
    results_text = "Predictive Maintenance Models\n"
    results_text += "=" * 50 + "\n\n"
    
    # Create a scaler (optional but useful for the web app)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save the scaler
    with open('failure_type_results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # First split dataset into train+validation and test sets (80% vs 20%)
    # Then split train+validation into train and validation (75% vs 25%, resulting in 60%/20%/20% overall)
    
    # Train models for each failure type and the combined model
    for failure_type in failure_modes + ['combined']:
        print(f"\n--- {failure_type} Failure Prediction Model ---")
        
        if failure_type == 'combined':
            y = df['Machine failure']
        else:
            y = df[failure_type]
        
        # First split: separate out the test set (20%)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Second split: create training (60% of total) and validation (20% of total) sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
        )
        
        print(f"Training set: {X_train.shape[0]} samples, Positive cases: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"Validation set: {X_val.shape[0]} samples, Positive cases: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
        print(f"Test set: {X_test.shape[0]} samples, Positive cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
        
        # Train model
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        # Validation performance
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Calculate ROC curve and AUC for validation
        val_fpr, val_tpr, _ = roc_curve(y_val, val_pred_proba)
        val_auc = auc(val_fpr, val_tpr)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation AUC-ROC: {val_auc:.4f}")
        
        # Save the model
        with open(f'failure_type_results/{failure_type.lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nClassification Report for {failure_type}:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix for {failure_type}:")
        print(cm)
        print(f"Test AUC-ROC: {roc_auc:.4f}")
        
        # Get feature importance
        feature_importances = model.feature_importances_
        top_features = sorted(list(zip(features, feature_importances)), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 Features for {}:".format(failure_type))
        for feature, importance in top_features[:5]:
            print(f"  {feature}: {importance:.4f}")
        
        # Save results to text
        results_text += f"--- {failure_type} Failure Prediction Model ---\n"
        results_text += f"Training set: {X_train.shape[0]} samples (60%)\n"
        results_text += f"Validation set: {X_val.shape[0]} samples (20%)\n"
        results_text += f"Test set: {X_test.shape[0]} samples (20%)\n\n"
        results_text += f"Validation Accuracy: {val_accuracy:.4f}\n"
        results_text += f"Validation AUC-ROC: {val_auc:.4f}\n\n"
        results_text += f"Test Accuracy: {test_accuracy:.4f}\n"
        results_text += f"Test AUC-ROC: {roc_auc:.4f}\n\n"
        results_text += f"Confusion Matrix (Test):\n"
        results_text += f"TN: {cm[0][0]}, FP: {cm[0][1]}\n"
        results_text += f"FN: {cm[1][0]}, TP: {cm[1][1]}\n\n"
        results_text += f"Classification Report (Test):\n"
        results_text += classification_report(y_test, y_pred)
        results_text += "\n"
        
        results_text += "Top 5 Feature Importance:\n"
        for feature, importance in top_features[:5]:
            results_text += f"- {feature}: {importance:.4f}\n"
        results_text += "\n\n"
        
        # Add model to results
        models[failure_type] = model
    
    # Add scaler to models
    models['scaler'] = scaler
    
    # Save results to file
    with open('failure_type_results/failure_type_models.txt', 'w') as f:
        f.write(results_text)
    
    print(f"\nResults saved to 'failure_type_results/failure_type_models.txt'")
    print("Failure type models training and evaluation complete!")
    
    return models

def train_models():
    """Train models but don't overwrite existing ones if they exist"""
    model_exists = os.path.exists('failure_type_results/combined_model.pkl')
    
    if model_exists:
        # Load existing models
        models = {}
        for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'combined']:
            model_path = f'failure_type_results/{failure_type.lower()}_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[failure_type] = pickle.load(f)
        
        # Load scaler
        scaler_path = 'failure_type_results/scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                models['scaler'] = pickle.load(f)
        
        return models
    else:
        # Train models from scratch
        return train_and_save_models()

if __name__ == "__main__":
    models = train_and_save_models() 