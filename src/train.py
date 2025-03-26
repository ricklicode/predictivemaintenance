import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Import custom modules
from data.dataset import PredictiveMaintenanceDataset
from models.predictor import PredictiveMaintenanceModel
from utils.evaluation import (plot_confusion_matrix, print_classification_report,
                            plot_roc_curve, plot_precision_recall_curve,
                            plot_training_history, plot_feature_importance)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a predictive maintenance model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='uc_pred_mait_ds.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--target', type=str, default='Machine failure',
                        help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data to use for validation')
    
    # Model parameters
    parser.add_argument('--hidden_layers', type=str, default='128,64,32',
                        help='Comma-separated list of neurons in hidden layers')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--multi_label', action='store_true',
                        help='If set, train a multi-label model for all failure types')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs to train for')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()

def train_model(args):
    """Train and evaluate the predictive maintenance model"""
    # Set random seed for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse hidden layers
    hidden_layers = [int(units) for units in args.hidden_layers.split(',')]
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load and preprocess the data
    print(f"Loading data from {args.data_path}")
    dataset = PredictiveMaintenanceDataset(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Get input shape for the model
    input_shape = dataset.get_input_shape()
    feature_names = dataset.get_feature_names()
    
    # Create and build the model
    print("Building model")
    model_args = {
        'input_shape': input_shape,
        'num_classes': 5 if args.multi_label else 1,  # 5 failure modes for multi-label
        'model_dir': args.model_dir
    }
    
    model = PredictiveMaintenanceModel(**model_args)
    
    model.build_model(
        hidden_layers=hidden_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        multi_label=args.multi_label
    )
    
    # Print model summary
    model.model.summary()
    
    # Train the model
    print("Training model")
    if args.multi_label:
        history = model.train(
            dataset.train_multi_dataset,
            dataset.val_multi_dataset,
            epochs=args.epochs,
            patience=args.patience
        )
    else:
        history = model.train(
            dataset.train_dataset,
            dataset.val_dataset,
            epochs=args.epochs,
            patience=args.patience
        )
    
    # Plot training history
    history_plot = plot_training_history(history)
    history_plot.savefig(os.path.join(run_dir, 'training_history.png'))
    
    # Evaluate the model
    print("Evaluating model")
    if args.multi_label:
        metrics = model.evaluate(dataset.test_multi_dataset)
    else:
        metrics = model.evaluate(dataset.test_dataset)
    
    # Print evaluation metrics
    print("Evaluation results:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save metrics to a file
    with open(os.path.join(run_dir, 'metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    # Generate predictions
    if args.multi_label:
        y_pred = model.predict(dataset.X_test)
        y_true = dataset.y_multi_test
    else:
        y_pred = model.predict(dataset.X_test)
        y_true = dataset.y_test
    
    # Generate evaluation plots for binary classification
    if not args.multi_label:
        # Confusion matrix
        cm_plot = plot_confusion_matrix(y_true, y_pred)
        cm_plot.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
        
        # Classification report
        print("Classification Report:")
        print_classification_report(y_true, y_pred)
        with open(os.path.join(run_dir, 'classification_report.txt'), 'w') as f:
            report = print_classification_report(y_true, y_pred)
            f.write(report if report else "")
        
        # ROC curve
        roc_plot, roc_auc = plot_roc_curve(y_true, y_pred)
        roc_plot.savefig(os.path.join(run_dir, 'roc_curve.png'))
        
        # Precision-recall curve
        pr_plot, pr_auc = plot_precision_recall_curve(y_true, y_pred)
        pr_plot.savefig(os.path.join(run_dir, 'precision_recall_curve.png'))
    else:
        # For multi-label classification, plot individual metrics for each failure mode
        failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for i, mode in enumerate(failure_modes):
            y_true_mode = y_true.iloc[:, i].values
            y_pred_mode = y_pred[:, i]
            
            # Confusion matrix
            cm_plot = plot_confusion_matrix(y_true_mode, y_pred_mode, 
                                          class_names=[f'No {mode}', mode])
            cm_plot.savefig(os.path.join(run_dir, f'confusion_matrix_{mode}.png'))
            
            # ROC curve
            roc_plot, _ = plot_roc_curve(y_true_mode, y_pred_mode)
            roc_plot.savefig(os.path.join(run_dir, f'roc_curve_{mode}.png'))
            
            # Precision-recall curve
            pr_plot, _ = plot_precision_recall_curve(y_true_mode, y_pred_mode)
            pr_plot.savefig(os.path.join(run_dir, f'precision_recall_curve_{mode}.png'))
    
    # Feature importance
    importance_plot, importance_df = plot_feature_importance(model.model, feature_names)
    importance_plot.savefig(os.path.join(run_dir, 'feature_importance.png'))
    importance_df.to_csv(os.path.join(run_dir, 'feature_importance.csv'), index=False)
    
    # Save the model
    model.save(os.path.join(run_dir, 'final_model.h5'))
    
    print(f"Training completed. Results saved to {run_dir}")
    
    return model, metrics

if __name__ == "__main__":
    args = parse_args()
    train_model(args) 