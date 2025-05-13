# Predictive Maintenance Model

A flexible and optimized machine learning model for predictive maintenance tasks. This project provides a generalized solution that can work with any dataset containing numeric features and a binary failure column.

## Features

- **Flexible Configuration**: Configure the model through a JSON file to work with any dataset
- **Advanced Feature Engineering**: Automatic creation of interaction features using optimized numpy operations
- **Optimized Training**: Fast model training with reduced hyperparameter search space and early stopping
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Interactive Dashboard**: Web-based dashboard to visualize model results and make predictions
- **Cross-Validation**: 5-fold stratified cross-validation for reliable performance estimation
- **Training History Tracking**: Monitor model performance during training
- **Feature Correlation Analysis**: Understand relationships between features
- **Failure Timeline Visualization**: Compare actual vs predicted failures

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictivemaintenance
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Your dataset should contain:
- Numeric features
- A binary failure column
- (Optional) An ID column

### 2. Configure the Model

Create a `model_config.json` file:
```json
{
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
```

### 3. Train the Model

```python
from generalized_model import PredictiveMaintenanceModel

# Initialize model with configuration
model = PredictiveMaintenanceModel('model_config.json')

# Load your data
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Train the model
results = model.fit(df)
```

### 4. Make Predictions

```python
# Make predictions on new data
new_data = df.sample(10)  # Example: predict on 10 random samples
predictions, probabilities = model.predict(new_data)
```

### 5. View Results

Run the web dashboard:
```bash
python app.py
```

Visit `http://localhost:5000` to view:
- Model performance metrics (Accuracy, ROC AUC, F1 Score)
- Feature importance analysis
- ROC and Precision-Recall curves
- Confusion Matrix
- Training History plots
- Feature Correlation Heatmap
- Failure Timeline visualization
- Model parameters

## Model Details

### Feature Engineering
- Automatic creation of interaction features
- Optimized using numpy operations for better performance
- Handles missing values and scaling
- Feature correlation analysis

### Training Process
- 5-fold stratified cross-validation
- Optimized hyperparameter search
- Early stopping for faster training
- Parallel processing for improved speed
- Training history tracking

### Performance Metrics
- Accuracy
- ROC AUC
- Average Precision
- F1 Score
- Feature Importance
- Training Time
- Confusion Matrix
- Feature Correlations

### Visualizations
- Training History plots (accuracy and loss)
- Feature Correlation Heatmap
- Failure Timeline (actual vs predicted)
- ROC and Precision-Recall curves
- Feature Importance plots
- Confusion Matrix
- Feature Distributions

## Project Structure

```
predictivemaintenance/
├── generalized_model.py    # Main model implementation
├── app.py                 # Web dashboard
├── model_config.json      # Model configuration
├── requirements.txt       # Project dependencies
├── model_results/         # Saved model and results
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_importance.csv
│   ├── results.txt
│   ├── training_history.json
│   ├── predictions.json
│   └── visualizations/
└── templates/            # Web dashboard templates
    └── index.html
```

## Performance Optimization

The model has been optimized for speed and efficiency:
1. **Feature Engineering**: Uses numpy operations instead of pandas
2. **Hyperparameter Search**: Reduced search space for faster training
3. **Early Stopping**: Starts from optimized default parameters
4. **Parallel Processing**: Utilizes multiple CPU cores
5. **Memory Efficiency**: Optimized data structures and operations
6. **Training History**: Efficient tracking of model performance
7. **Visualization Optimization**: Optimized plot generation and caching

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
