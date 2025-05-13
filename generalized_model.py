#!/usr/bin/env python
"""
Generalized Predictive Maintenance Model
This script provides a flexible implementation for predictive maintenance tasks
that can work with any dataset containing numeric features and a binary failure column.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, log_loss
)
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model with optional configuration
        
        Parameters:
        -----------
        config_path : str, optional
            Path to a JSON configuration file containing:
            - feature_columns: list of feature column names
            - failure_column: name of the failure column
            - id_column: name of the unique ID column
            - output_dir: directory for saving results
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.permutation_importance = None
        self.feature_columns = None
        self.failure_column = None
        self.id_column = None
        self.output_dir = self.config.get('output_dir', 'model_results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features based on available numeric columns
        Optimized version using numpy operations
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with additional engineered features
        """
        df_eng = df.copy()
        
        # Get numeric columns for feature engineering
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        
        # Remove ID and failure columns if they exist
        if self.id_column and self.id_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.id_column)
        if self.failure_column and self.failure_column in numeric_cols:
            numeric_cols = numeric_cols.drop(self.failure_column)
        
        # Create a dictionary to store new features
        new_features = {}
        
        # Create interaction features using numpy operations for better performance
        n = len(numeric_cols)
        for i in range(n):
            for j in range(i+1, n):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                # Use numpy operations for better performance
                new_features[f'{col1}_{col2}_product'] = np.multiply(df_eng[col1], df_eng[col2])
                new_features[f'{col1}_{col2}_ratio'] = np.divide(df_eng[col1], df_eng[col2] + 1e-6)
                new_features[f'{col1}_{col2}_diff'] = np.subtract(df_eng[col1], df_eng[col2])
        
        # Add all new features at once using pd.concat
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df_eng.index)
            df_eng = pd.concat([df_eng, new_features_df], axis=1)
        
        return df_eng
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with prepared features
        """
        # Get feature columns from config or use all numeric columns
        if self.feature_columns:
            feature_cols = self.feature_columns
        else:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID and failure columns if they exist
            if self.id_column and self.id_column in feature_cols:
                feature_cols.remove(self.id_column)
            if self.failure_column and self.failure_column in feature_cols:
                feature_cols.remove(self.failure_column)
        
        return df[feature_cols]
    
    def fit(self, df: pd.DataFrame, failure_column: Optional[str] = None,
            id_column: Optional[str] = None) -> Dict:
        """
        Train the model on the provided dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing features and failure column
        failure_column : str, optional
            Name of the failure column (overrides config)
        id_column : str, optional
            Name of the ID column (overrides config)
            
        Returns:
        --------
        dict
            Dictionary containing training results and metrics
        """
        # Start timing
        start_time = time.time()
        
        # Update configuration
        if failure_column:
            self.failure_column = failure_column
        elif 'failure_column' in self.config:
            self.failure_column = self.config['failure_column']
        else:
            raise ValueError("Failure column must be specified either in config or as parameter")
            
        if id_column:
            self.id_column = id_column
        elif 'id_column' in self.config:
            self.id_column = self.config['id_column']
            
        if 'feature_columns' in self.config:
            self.feature_columns = self.config['feature_columns']
        
        # Engineer features
        logger.info("Engineering features...")
        df_eng = self._engineer_features(df)
        
        # Prepare features and target
        X = self._prepare_features(df_eng)
        y = df[self.failure_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define optimized hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],  # Reduced from 3 to 2 options
            'max_depth': [None, 20],     # Reduced from 4 to 2 options
            'min_samples_split': [2, 5], # Reduced from 3 to 2 options
            'min_samples_leaf': [1, 2],  # Reduced from 3 to 2 options
            'max_features': ['sqrt'],    # Reduced from 2 to 1 option
            'class_weight': ['balanced'] # Reduced from 2 to 1 option
        }
        
        # Initialize base model with early stopping parameters
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced'
        )
        
        # Initialize and train model with optimized parameters
        logger.info("Training model...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Initialize training history
        history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': []
        }
        
        # Train model with history tracking
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train model
            self.model = grid_search.fit(X_fold_train, y_fold_train).best_estimator_
            
            # Record metrics
            train_pred = self.model.predict(X_fold_train)
            val_pred = self.model.predict(X_fold_val)
            
            history['train_accuracy'].append(accuracy_score(y_fold_train, train_pred))
            history['val_accuracy'].append(accuracy_score(y_fold_val, val_pred))
            history['train_loss'].append(log_loss(y_fold_train, train_pred))
            history['val_loss'].append(log_loss(y_fold_val, val_pred))
        
        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        # Save predictions
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]
        with open(os.path.join(self.output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions.tolist(), f)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, self.model.predict(X_test_scaled))
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        avg_precision = average_precision_score(y_test, predictions)
        fpr, tpr, _ = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_test, self.model.predict(X_test_scaled))
        cm = confusion_matrix(y_test, self.model.predict(X_test_scaled))
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Calculate permutation importance with reduced repeats
        result = permutation_importance(
            self.model, X_test_scaled, y_test, 
            n_repeats=5,  # Reduced from 10 to 5
            random_state=42, 
            n_jobs=-1
        )
        self.permutation_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Save confusion matrix as image and JSON
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png')
        plt.close()
        with open(f'{self.output_dir}/confusion_matrix.json', 'w') as f:
            json.dump({'confusion_matrix': cm.tolist()}, f)

        # Save feature distributions (histogram data)
        feature_distributions = {}
        for col in X.columns:
            counts, bin_edges = np.histogram(X[col], bins=20)
            feature_distributions[col] = {
                'counts': counts.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        with open(f'{self.output_dir}/feature_distributions.json', 'w') as f:
            json.dump(feature_distributions, f)

        # Save results
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'f1': f1,
            'best_params': grid_search.best_params_,
            'feature_importance': self.feature_importance.to_dict(),
            'permutation_importance': self.permutation_importance.to_dict(),
            'training_time': time.time() - start_time,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'confusion_matrix': cm.tolist(),
            'feature_distributions': feature_distributions
        }
        
        # Save model and results
        self._save_results(results)
        
        return results
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing features
            
        Returns:
        --------
        tuple
            (predictions, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Engineer features
        df_eng = self._engineer_features(df)
        
        # Prepare features
        X = self._prepare_features(df_eng)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        prediction_probas = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, prediction_probas
    
    def _save_results(self, results: Dict) -> None:
        """
        Save model, results, and visualizations
        
        Parameters:
        -----------
        results : dict
            Dictionary containing training results and metrics
        """
        # Save model
        with open(f'{self.output_dir}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(f'{self.output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature importance
        self.feature_importance.to_csv(f'{self.output_dir}/feature_importance.csv', index=False)
        
        # Save ROC curve data
        with open(f'{self.output_dir}/roc_data.json', 'w') as f:
            json.dump({
                'fpr': results['fpr'],
                'tpr': results['tpr']
            }, f)
        
        # Save PR curve data
        with open(f'{self.output_dir}/pr_data.json', 'w') as f:
            json.dump({
                'precision': results['precision'],
                'recall': results['recall']
            }, f)
        
        # Save F1 and confusion matrix in results.txt
        with open(f'{self.output_dir}/results.txt', 'a') as f:
            f.write(f"F1 Score: {results['f1']:.4f}\n")
            f.write("Confusion Matrix:\n")
            for row in results['confusion_matrix']:
                f.write(f"  {row}\n")
        
        # Create visualizations
        self._create_visualizations(results)
    
    def _create_visualizations(self, results: Dict) -> None:
        """
        Create and save visualizations
        
        Parameters:
        -----------
        results : dict
            Dictionary containing training results and metrics
        """
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=self.feature_importance.head(10))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png')
        plt.close()
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], 
                label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.output_dir}/roc_curve.png')
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(results['recall'], results['precision'], 
                label=f'PR curve (AP = {results["avg_precision"]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(f'{self.output_dir}/pr_curve.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "feature_columns": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ],
        "failure_column": "Machine failure",
        "id_column": "UID",
        "output_dir": "model_results"
    }
    
    # Save config
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data
    df = pd.read_csv('uc_pred_mait_ds.csv')
    
    # Initialize and train model
    model = PredictiveMaintenanceModel('model_config.json')
    results = model.fit(df)
    
    # Make predictions on new data
    new_data = df.sample(10)  # Example: predict on 10 random samples
    predictions, probabilities = model.predict(new_data)
    
    print("\nPredictions on new data:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Prediction = {pred}, Probability = {prob:.4f}") 