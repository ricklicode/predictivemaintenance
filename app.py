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
    """Load model results and metrics"""
    results = {}
    
    # Load results text file
    with open('model_results/results.txt', 'r') as f:
        results['text'] = f.read()
    
    # Load model and scaler
    with open('model_results/model.pkl', 'rb') as f:
        results['model'] = pickle.load(f)
    with open('model_results/scaler.pkl', 'rb') as f:
        results['scaler'] = pickle.load(f)
    
    # Load feature importance
    feature_importance = pd.read_csv('model_results/feature_importance.csv')
    results['feature_importance'] = feature_importance.to_dict('records')
    
    # Load ROC curve data
    with open('model_results/roc_data.json', 'r') as f:
        results['roc_data'] = json.load(f)
    
    # Load PR curve data
    with open('model_results/pr_data.json', 'r') as f:
        results['pr_data'] = json.load(f)
    
    return results

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
    
    # Create feature distributions plot
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(['Air temperature [K]', 'Process temperature [K]', 
                                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']):
        plt.subplot(2, 3, i+1)
        sns.histplot(dataset[feature], kde=True)
        plt.title(feature)
    plt.tight_layout()
    dist_plot = create_plot(plt.gcf())
    plt.close()
    
    # Create failure type distribution
    plt.figure(figsize=(10, 6))
    failure_counts = [stats['failure_types'][t] for t in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    plt.bar(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], failure_counts)
    plt.title('Failure Type Distribution')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    failures_plot = create_plot(plt.gcf())
    plt.close()
    
    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    corr_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                     'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
    corr_matrix = dataset[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    corr_plot = create_plot(plt.gcf())
    plt.close()
    
    # Load model results
    try:
        results = load_model_results()
        model_results = results['text']
        failure_model_results = "Model results not available."
    except:
        model_results = "Model results not available."
    
    return render_template('index.html', 
                          stats=stats,
                          dist_plot=dist_plot,
                          failures_plot=failures_plot,
                          corr_plot=corr_plot,
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
    
    app.run(debug=True) 