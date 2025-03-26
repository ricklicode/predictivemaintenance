#!/usr/bin/env python
"""
Command-line script to train a predictive maintenance model.
Example usage:
    python run_training.py --epochs 30 --batch_size 64
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the training function
from src.train import parse_args, train_model

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Train the model
    model, metrics = train_model(args)
    
    # Print completion message
    print("\nTraining completed successfully!")
    print(f"Model saved to {os.path.join(args.model_dir, 'final_model.h5')}")
    
    # Print key metrics
    print("\nKey metrics:")
    for metric_name in ['loss', 'accuracy', 'auc', 'precision', 'recall']:
        if metric_name in metrics:
            print(f"  {metric_name}: {metrics[metric_name]:.4f}") 