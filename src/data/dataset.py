import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PredictiveMaintenanceDataset:
    """
    A custom dataset class for the UC Predictive Maintenance dataset.
    This class preprocesses the data and provides TensorFlow compatible datasets
    for training, validation, and testing.
    """
    
    def __init__(self, data_path, target='Machine failure', test_size=0.2, val_size=0.1, 
                 batch_size=32, seed=42):
        """
        Initialize the dataset class.
        
        Args:
            data_path (str): Path to the CSV file
            target (str): Target column name
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            batch_size (int): Batch size for training
            seed (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.seed = seed
        self.scaler = StandardScaler()
        
        # Load and preprocess the data
        self._load_data()
        self._preprocess_data()
        self._split_data()
        self._create_tf_datasets()
        
    def _load_data(self):
        """Load the dataset from CSV"""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with {self.df.shape[0]} samples and {self.df.shape[1]} features")
        
    def _preprocess_data(self):
        """Preprocess the dataset"""
        # Drop non-numeric columns that aren't needed for prediction
        self.df_processed = self.df.copy()
        
        # Extract the product type from Product ID (L, M, H)
        self.df_processed['Product_Type'] = self.df_processed['Product ID'].str[0]
        
        # One-hot encode the product type
        product_type_dummies = pd.get_dummies(self.df_processed['Product_Type'], 
                                             prefix='Product_Type')
        self.df_processed = pd.concat([self.df_processed, product_type_dummies], axis=1)
        
        # Drop columns that aren't used for training
        drop_cols = ['UDI', 'Product ID', 'Product_Type']
        self.df_processed.drop(drop_cols, axis=1, inplace=True)
        
        # Define features and target
        self.y = self.df_processed[self.target]
        
        # Also create multi-target for individual failure modes if we want to use them
        self.y_multi = self.df_processed[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
        # Drop target columns from features
        self.X = self.df_processed.drop([self.target, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
        
        # Scale the numeric features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Preprocessed data: {self.X_scaled.shape[0]} samples with {self.X_scaled.shape[1]} features")
        
    def _split_data(self):
        """Split the data into training, validation, and test sets"""
        # First, split into training+validation and test sets
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=self.test_size, random_state=self.seed, stratify=self.y
        )
        
        # Then split training set into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size/(1-self.test_size), 
            random_state=self.seed, stratify=y_train_val
        )
        
        # Also split the multi-target data
        _, self.y_multi_test, y_multi_train_val, _ = train_test_split(
            self.X_scaled, self.y_multi, test_size=self.test_size, 
            random_state=self.seed, stratify=self.y
        )
        
        self.y_multi_train, self.y_multi_val, _, _ = train_test_split(
            y_multi_train_val, y_train_val, test_size=self.val_size/(1-self.test_size), 
            random_state=self.seed, stratify=y_train_val
        )
        
        print(f"Data split: Train={self.X_train.shape[0]}, Val={self.X_val.shape[0]}, Test={self.X_test.shape[0]}")
        
    def _create_tf_datasets(self):
        """Create TensorFlow datasets for training, validation, and testing"""
        # Create training dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train)
        ).shuffle(buffer_size=len(self.X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val, self.y_val)
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Create test dataset
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_test, self.y_test)
        ).batch(self.batch_size)
        
        # Create multi-target datasets if needed
        self.train_multi_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_multi_train)
        ).shuffle(buffer_size=len(self.X_train)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.val_multi_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val, self.y_multi_val)
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.test_multi_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_test, self.y_multi_test)
        ).batch(self.batch_size)
        
    def get_feature_names(self):
        """Return the feature names"""
        return list(self.X.columns)
    
    def get_input_shape(self):
        """Return the input shape for the model"""
        return self.X_train.shape[1:] 