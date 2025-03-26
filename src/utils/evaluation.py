import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues'):
    """
    Plot confusion matrix for binary or multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (probabilities will be converted to binary)
        class_names: List of class names
        figsize: Figure size tuple
        cmap: Colormap for the plot
    """
    # Convert probabilities to binary predictions if needed
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multi-class case
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        # Binary case
        y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=class_names if class_names else ['Negative', 'Positive'],
                yticklabels=class_names if class_names else ['Negative', 'Positive'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Calculate and display metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return plt.gcf()

def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (probabilities will be converted to binary)
        target_names: List of target class names
    """
    # Convert probabilities to binary predictions if needed
    if isinstance(y_pred[0], (list, np.ndarray)):
        y_pred_classes = (y_pred > 0.5).astype(int)
    else:
        y_pred_classes = (np.array(y_pred) > 0.5).astype(int)
    
    # Print classification report
    report = classification_report(y_true, y_pred_classes, target_names=target_names)
    print(report)
    
def plot_roc_curve(y_true, y_pred_proba, figsize=(10, 8)):
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size tuple
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    return plt.gcf(), roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(10, 8)):
    """
    Plot Precision-Recall curve and calculate AUC.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size tuple
    """
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    return plt.gcf(), pr_auc

def plot_training_history(history, figsize=(15, 6)):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras history object from model.fit()
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=10, figsize=(12, 8)):
    """
    For a trained TensorFlow model, use a permutation approach to estimate
    feature importance and plot the results.
    
    Args:
        model: Trained TensorFlow model
        feature_names: List of feature names
        X_val: Validation features
        y_val: Validation targets
        top_n: Number of top features to display
        figsize: Figure size tuple
    """
    # This is a placeholder for a feature importance function
    # In a real implementation, you would use permutation importance
    # or other methods to calculate feature importance
    
    # For demonstration, we'll just use the first layer weights as a proxy
    # This is not accurate for complex models but serves as an example
    
    # Get the weights from the first layer
    weights = model.layers[1].get_weights()[0]
    
    # Calculate absolute importance for each feature
    importance = np.mean(np.abs(weights), axis=1)
    
    # Create a dataframe for better visualization
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return plt.gcf(), importance_df 