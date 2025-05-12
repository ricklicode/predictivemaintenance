#!/usr/bin/env python
"""
Script to run the generalized predictive maintenance model on the UC dataset
"""

import pandas as pd
from generalized_model import PredictiveMaintenanceModel
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load the dataset
    logger.info("Loading dataset...")
    df = pd.read_csv('uc_pred_mait_ds.csv')
    logger.info(f"Dataset shape: {df.shape}")
    
    # Initialize model with configuration
    logger.info("Initializing model...")
    model = PredictiveMaintenanceModel('model_config.json')
    
    # Train the model
    logger.info("Training model...")
    results = model.fit(df)
    
    # Print results
    logger.info("\nModel Results:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
    logger.info(f"Average Precision: {results['avg_precision']:.4f}")
    logger.info(f"Training Time: {results['training_time']:.2f} seconds")
    
    logger.info("\nBest Parameters:")
    for param, value in results['best_params'].items():
        logger.info(f"  {param}: {value}")
    
    # Make predictions on a sample of the data
    logger.info("\nMaking predictions on sample data...")
    sample_data = df.sample(5)  # Get 5 random samples
    predictions, probabilities = model.predict(sample_data)
    
    logger.info("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        logger.info(f"Sample {i+1}: Prediction = {pred}, Probability = {prob:.4f}")
    
    logger.info("\nResults and visualizations have been saved to the 'model_results' directory")

if __name__ == "__main__":
    main() 