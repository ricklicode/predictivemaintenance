# Predictive Maintenance System

This project implements a machine learning solution for predictive maintenance using scikit-learn. It includes scripts for data analysis, model training, and a web application for interactive predictions.

## Dataset

The project uses the UC Predictive Maintenance dataset with 10,000 data points and the following features:
- UID: unique identifier 
- Product ID: product quality variants (L, M, H) with serial numbers
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine failure (target label)

The machine failures can be due to five different modes:
- Tool wear failure (TWF)
- Heat dissipation failure (HDF)
- Power failure (PWF)
- Overstrain failure (OSF)
- Random failures (RNF)

## Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── uc_pred_mait_ds.csv                # Dataset file
├── app.py                             # Web application
├── simple_model.py                    # Model training script
├── simple_analysis.py                 # Dataset analysis script
├── failure_type_model.py              # Failure type prediction models
├── failure_type_results/              # Failure type model results
│   ├── failure_type_models.txt        # Detailed model metrics
│   ├── combined_model.pkl             # Combined failure prediction model
│   ├── twf_model.pkl                  # Tool wear failure model
│   ├── hdf_model.pkl                  # Heat dissipation failure model
│   ├── pwf_model.pkl                  # Power failure model
│   ├── osf_model.pkl                  # Overstrain failure model
│   ├── rnf_model.pkl                  # Random failure model
│   └── scaler.pkl                     # Feature scaler
├── model_results/                     # Combined model results
│   └── model_results.txt              # Model performance metrics
├── simple_results/                    # Analysis results
│   └── dataset_analysis.txt           # Dataset statistics
└── templates/                         # Web app templates
    └── index.html                     # Dashboard UI template
```

## Setup and Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the dataset file `uc_pred_mait_ds.csv` in the project root directory.

## Running the Data Analysis

To analyze the dataset and generate statistics:

```bash
python simple_analysis.py
```

This will create a `simple_results` directory with analysis results.

## Training Models

To train the combined machine failure prediction model:

```bash
python simple_model.py
```

To train models for each failure type:

```bash
python failure_type_model.py
```

The trained models will be saved in the `failure_type_results` directory.

## Web Application

The project includes a web-based dashboard for visualizing results and making predictions.

### Starting the Web App

To launch the web application:

```bash
python app.py
```

This will start a Flask server, typically at http://127.0.0.1:5000/

### Accessing the Dashboard

1. Open your web browser and go to http://127.0.0.1:5000/
2. You'll see the main dashboard with dataset statistics, visualizations, and the prediction interface.

### Using the Prediction Interface

The web application allows you to make predictions with the trained models:

1. Navigate to the "Make a Prediction" section of the dashboard.
2. Enter values for the machine parameters:
   - Air Temperature [K]
   - Process Temperature [K]
   - Rotational Speed [rpm]
   - Torque [Nm]
   - Tool Wear [min]
   - Product Type (L, M, or H)
3. Click the "Predict" button to get failure predictions.
4. The results will show:
   - Overall machine failure probability
   - Individual probabilities for each failure type (TWF, HDF, PWF, OSF, RNF)

### Training Models Through the Web Interface

If you need to retrain models:

1. Click the "Train Models" button in the prediction form.
2. Wait for the training to complete (this may take a few minutes).
3. The page will refresh with updated model information.

## Model Training Details

The models are trained with a proper data splitting strategy:

1. **Training Set (60% of data)**: Used to train the models
2. **Validation Set (20% of data)**: Used to monitor performance during development
3. **Test Set (20% of data)**: Used for final evaluation

All splits are stratified to maintain the same class distribution, and use fixed random seeds for reproducibility.

## Model Performance

The performance metrics for each model are available in `failure_type_results/failure_type_models.txt`. Key highlights:

- High accuracy across all models (>99%)
- AUC-ROC values near 1.0 for most failure types
- Power Failure (PWF) prediction achieves perfect accuracy
- The combined model achieves 99.25% test accuracy with 0.95+ AUC-ROC

## License

This project is open-source and available under the MIT License.
