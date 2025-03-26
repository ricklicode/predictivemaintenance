import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os

class PredictiveMaintenanceModel:
    """
    A class for building and training the predictive maintenance model.
    """
    
    def __init__(self, input_shape, num_classes=1, model_dir='./saved_models'):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of the input features
            num_classes (int): Number of output classes (1 for binary, >1 for multi-class)
            model_dir (str): Directory to save model checkpoints
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self, hidden_layers=[64, 32], dropout_rate=0.3, 
                   learning_rate=0.001, multi_label=False):
        """
        Build the neural network model.
        
        Args:
            hidden_layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the optimizer
            multi_label (bool): Whether this is a multi-label classification problem
            
        Returns:
            The compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Hidden layers
        x = inputs
        for units in hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        if multi_label:
            # Multiple outputs for multi-label classification
            outputs = Dense(self.num_classes, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            # Single output for binary classification
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, train_dataset, val_dataset, epochs=50, patience=10):
        """
        Train the model.
        
        Args:
            train_dataset: TensorFlow dataset for training
            val_dataset: TensorFlow dataset for validation
            epochs (int): Maximum number of epochs to train for
            patience (int): Patience for early stopping
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on the test dataset.
        
        Args:
            test_dataset: TensorFlow dataset for testing
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Evaluate the model
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Create a dictionary of metrics
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X: Input features (numpy array or TensorFlow dataset)
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Make predictions
        return self.model.predict(X)
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model to. If None, use default location.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'final_model.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath=None):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model. If None, use default location.
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'best_model.h5')
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model 