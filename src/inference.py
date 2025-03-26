import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions with a trained predictive maintenance model')
    
    # Input parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data for prediction (CSV format)')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save the predictions')
    parser.add_argument('--multi_label', action='store_true',
                        help='Whether the model is multi-label')
    
    return parser.parse_args()

def preprocess_data(data_path):
    """
    Preprocess the data for inference.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Preprocessed data ready for prediction
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Extract the product type from Product ID (L, M, H)
    df_processed['Product_Type'] = df_processed['Product ID'].str[0]
    
    # One-hot encode the product type
    product_type_dummies = pd.get_dummies(df_processed['Product_Type'], 
                                         prefix='Product_Type')
    df_processed = pd.concat([df_processed, product_type_dummies], axis=1)
    
    # Drop columns that aren't used for training
    drop_cols = ['UDI', 'Product ID', 'Product_Type']
    
    # If target columns exist, drop them too
    target_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in target_cols:
        if col in df_processed.columns:
            drop_cols.append(col)
    
    df_processed = df_processed.drop(drop_cols, axis=1, inplace=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed)
    
    return X_scaled, df

def load_model(model_path):
    """
    Load a trained TensorFlow model.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        The loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(model, X, multi_label=False):
    """
    Make predictions with the trained model.
    
    Args:
        model: Trained TensorFlow model
        X: Preprocessed input features
        multi_label: Whether this is a multi-label model
        
    Returns:
        Predictions
    """
    # Make predictions
    predictions = model.predict(X)
    
    if multi_label:
        # Format multi-label predictions
        failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        pred_df = pd.DataFrame(predictions, columns=failure_modes)
        # Also add a 'Machine failure' column for any type of failure
        pred_df['Machine failure'] = (pred_df > 0.5).any(axis=1).astype(int)
    else:
        # Format binary predictions
        pred_df = pd.DataFrame({'Machine failure probability': predictions.flatten()})
        pred_df['Machine failure'] = (predictions > 0.5).astype(int).flatten()
    
    return pred_df

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Preprocess data
    print(f"Preprocessing data from {args.data_path}")
    X, original_df = preprocess_data(args.data_path)
    
    # Make predictions
    print("Making predictions")
    predictions = make_predictions(model, X, args.multi_label)
    
    # Combine original data with predictions
    result = pd.concat([original_df, predictions], axis=1)
    
    # Save predictions
    result.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    # Print summary of predictions
    print("\nPrediction Summary:")
    if args.multi_label:
        for col in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            if col in predictions.columns:
                print(f"  {col}: {predictions[col].sum()} predicted failures out of {len(predictions)} samples")
    else:
        print(f"  Machine failures: {predictions['Machine failure'].sum()} predicted failures out of {len(predictions)} samples")
        print(f"  Average failure probability: {predictions['Machine failure probability'].mean():.4f}")

if __name__ == "__main__":
    main() 