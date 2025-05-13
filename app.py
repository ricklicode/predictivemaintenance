import os
import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import base64
from generalized_model import PredictiveMaintenanceModel

app = Flask(__name__)

# Load models
def load_models():
    models = {}
    # Load failure type models
    for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'combined']:
        model_path = f'failure_type_results/{failure_type.lower()}_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[failure_type] = pickle.load(f)
    
    # Load scaler if exists
    scaler_path = 'failure_type_results/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    
    return models

# Try to load models, if they don't exist we'll train them on first access
try:
    MODELS = load_models()
except:
    MODELS = {}

# Load dataset
def load_dataset():
    try:
        return pd.read_csv('uc_pred_mait_ds.csv')
    except:
        return None

# Create a DataFrame copy
def prepare_features(data):
    # Extract product types
    data['Product_Type'] = data['Product ID'].str[0]
    
    # Create dummy variables for product type
    product_dummies = pd.get_dummies(data['Product_Type'], prefix='Product_Type')
    data = pd.concat([data, product_dummies], axis=1)
    
    # Engineer additional features
    data['Power [W]'] = data['Torque [Nm]'] * data['Rotational speed [rpm]'] / 9.5488
    data['Temp_Diff [K]'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Tool_Torque [minNm]'] = data['Tool wear [min]'] * data['Torque [Nm]']
    
    # Select features for model
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                'Torque [Nm]', 'Tool wear [min]', 'Product_Type_L', 'Product_Type_M', 
                'Product_Type_H', 'Power [W]', 'Temp_Diff [K]', 'Tool_Torque [minNm]']
    
    return data[features]

# Train models if they don't exist
def train_models_if_needed():
    global MODELS
    if 'combined' not in MODELS:
        from simple_model import train_models
        MODELS = train_models()
    return MODELS

# Create plot image
def create_plot(plt_figure):
    img = io.BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def load_model_results():
    """Load model results and metrics in a frontend-friendly format"""
    results = {}

    # Parse metrics from results.txt
    metrics = {}
    best_params = {}
    with open('model_results/results.txt', 'r') as f:
        for line in f:
            if line.startswith('Accuracy:'):
                metrics['accuracy'] = float(line.split(':')[1].strip())
            elif line.startswith('ROC AUC:'):
                metrics['roc_auc'] = float(line.split(':')[1].strip())
            elif line.startswith('Average Precision:'):
                metrics['avg_precision'] = float(line.split(':')[1].strip())
            elif line.startswith('Training Time:'):
                metrics['training_time'] = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('F1 Score:'):
                metrics['f1'] = float(line.split(':')[1].strip())
            elif line.strip().startswith('Best Parameters:'):
                section = 'params'
            elif line.strip().startswith('Top 5 Features by Importance:'):
                section = 'features'
            elif 'section' in locals() and section == 'params' and ':' in line:
                param, value = line.strip().split(':', 1)
                best_params[param.strip()] = value.strip()
            elif 'section' in locals() and section == 'features' and ':' in line:
                # skip, handled by feature_importance.csv
                pass

    # Load feature importance
    feature_importance = pd.read_csv('model_results/feature_importance.csv')
    feature_importance_list = feature_importance.to_dict('records')

    # Load ROC curve data
    with open('model_results/roc_data.json', 'r') as f:
        roc_data = json.load(f)

    # Load PR curve data
    with open('model_results/pr_data.json', 'r') as f:
        pr_data = json.load(f)

    # Load confusion matrix
    with open('model_results/confusion_matrix.json', 'r') as f:
        confusion_matrix_data = json.load(f)['confusion_matrix']

    # Load feature distributions
    with open('model_results/feature_distributions.json', 'r') as f:
        feature_distributions = json.load(f)

    # Compose results for frontend
    results.update(metrics)
    results['feature_importance'] = feature_importance_list
    results['roc_data'] = roc_data
    results['pr_data'] = pr_data
    results['best_params'] = best_params
    results['confusion_matrix'] = confusion_matrix_data
    results['feature_distributions'] = feature_distributions
    return results

def create_training_history_plot():
    """Create a plot showing training and validation metrics over time"""
    plt.figure(figsize=(10, 6))
    
    # Load training history from model_results
    with open('model_results/training_history.json', 'r') as f:
        history = json.load(f)
    
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return create_plot(plt.gcf())

def create_feature_correlation_plot(dataset):
    """Create a heatmap showing correlations between all features"""
    plt.figure(figsize=(12, 10))
    
    # Select numeric features
    numeric_features = dataset.select_dtypes(include=[np.number]).columns
    corr_matrix = dataset[numeric_features].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': .8})
    
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return create_plot(plt.gcf())

def create_feature_distributions_plot(dataset):
    """Create distribution plots for each feature"""
    plt.figure(figsize=(15, 10))
    
    # Select numeric features
    numeric_features = dataset.select_dtypes(include=[np.number]).columns
    
    # Create subplots
    n_features = len(numeric_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Plot each feature
    for idx, feature in enumerate(numeric_features):
        sns.histplot(data=dataset, x=feature, ax=axes[idx], kde=True)
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].set_xlabel('')
    
    # Remove empty subplots
    for idx in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return create_plot(plt.gcf())

def create_failure_type_distribution_plot(stats):
    """Create a plot showing the distribution of failure types"""
    plt.figure(figsize=(10, 6))
    
    # Extract failure type counts
    failure_types = list(stats['failure_types'].keys())
    counts = list(stats['failure_types'].values())
    
    # Create bar plot
    plt.bar(failure_types, counts)
    plt.title('Distribution of Failure Types')
    plt.xlabel('Failure Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    return create_plot(plt.gcf())

# Routes
@app.route('/')
def index():
    """Render the main dashboard page"""
    # Make sure models are loaded
    models = train_models_if_needed()
    
    # Load dataset statistics
    dataset = load_dataset()
    if dataset is None:
        return "Dataset not found. Please upload the dataset first."
    
    # Basic dataset stats
    stats = {
        'total_samples': len(dataset),
        'machine_failures': dataset['Machine failure'].sum(),
        'failure_rate': dataset['Machine failure'].mean() * 100,
        'failure_types': {
            'TWF': dataset['TWF'].sum(),
            'HDF': dataset['HDF'].sum(), 
            'PWF': dataset['PWF'].sum(),
            'OSF': dataset['OSF'].sum(),
            'RNF': dataset['RNF'].sum()
        }
    }
    
    # Create all plots
    dist_plot = create_feature_distributions_plot(dataset)
    failures_plot = create_failure_type_distribution_plot(stats)
    corr_plot = create_feature_correlation_plot(dataset)
    training_history_plot = create_training_history_plot()
    
    # Load model results
    try:
        results = load_model_results()
        model_results = results.get('text', "Model results not available.")
        failure_model_results = "Model results not available."
    except:
        model_results = "Model results not available."
        failure_model_results = "Model results not available."
    
    return render_template('index.html', 
                          stats=stats,
                          dist_plot=dist_plot,
                          failures_plot=failures_plot,
                          corr_plot=corr_plot,
                          training_history_plot=training_history_plot,
                          model_results=model_results,
                          failure_model_results=failure_model_results)

@app.route('/api/results')
def get_results():
    """API endpoint to get model results"""
    try:
        results = load_model_results()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions"""
    try:
        data = request.json
        features = pd.DataFrame([data['features']])
        
        # Load model and scaler
        with open('model_results/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_results/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale features and make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Save models for the app to use
def save_models_for_app():
    from simple_model import train_and_save_models
    train_and_save_models()
    return "Models trained and saved successfully."

@app.route('/train_models', methods=['POST'])
def train_models_route():
    result = save_models_for_app()
    global MODELS
    MODELS = load_models()
    return result

if __name__ == '__main__':
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    
    # Check if models are available, train them if needed
    if not os.path.exists('failure_type_results'):
        os.makedirs('failure_type_results', exist_ok=True)
    
    app.run(debug=True, port=8080) 