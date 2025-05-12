import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import logging
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === Directory Structure Setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

# === Logger Setup ===
def setup_logger(name):
    log_file = os.path.join(LOG_DIR, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize main logger
logger = setup_logger('loan_prediction')

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.numeric_cols = None
        self.categorical_cols = None
        self.logger = setup_logger('feature_preprocessor')

    def fit(self, X):
        self.logger.info("Starting feature preprocessing fit")
        try:
            # Store column information
            self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
            
            self.logger.info(f"Detected numeric columns: {self.numeric_cols}")
            self.logger.info(f"Detected categorical columns: {self.categorical_cols}")
            
            # Remove target column if present
            if 'loanRisk' in self.numeric_cols:
                self.numeric_cols.remove('loanRisk')
                self.logger.info("Removed 'loanRisk' from numeric columns")
            if 'loanRisk' in self.categorical_cols:
                self.categorical_cols.remove('loanRisk')
                self.logger.info("Removed 'loanRisk' from categorical columns")
                
            # Fit scaler on numeric columns
            if self.numeric_cols:
                self.logger.info(f"Fitting StandardScaler on {len(self.numeric_cols)} numeric columns")
                self.scaler.fit(X[self.numeric_cols])
                self.logger.info(f"StandardScaler fitted successfully")
                
            # Fit encoders on categorical columns
            for col in self.categorical_cols:
                self.logger.info(f"Fitting LabelEncoder on column: {col}")
                le = LabelEncoder()
                unique_values = X[col].astype(str).unique()
                self.logger.info(f"Column {col} has {len(unique_values)} unique values")
                le.fit(X[col].astype(str))
                self.encoders[col] = le
                self.logger.info(f"LabelEncoder fitted successfully for column: {col}")
                
            self.logger.info("Feature preprocessing fit completed successfully")
            return self
        except Exception as e:
            self.logger.error(f"Error in fit method: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def transform(self, X):
        self.logger.info("Starting feature preprocessing transform")
        try:
            X = X.copy()
            self.logger.info(f"Input data shape: {X.shape}")
            
            # Remove target column if present
            if 'loanRisk' in X.columns:
                X = X.drop('loanRisk', axis=1)
                self.logger.info("Removed 'loanRisk' column from input data")
                
            # Transform numeric columns
            if self.numeric_cols:
                self.logger.info(f"Transforming {len(self.numeric_cols)} numeric columns")
                # Check if all numeric columns are present in the input data
                missing_cols = set(self.numeric_cols) - set(X.columns)
                if missing_cols:
                    self.logger.warning(f"Missing numeric columns in input data: {missing_cols}")
                    for col in missing_cols:
                        self.numeric_cols.remove(col)
                
                X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
                self.logger.info("Numeric columns transformed successfully")
                
            # Transform categorical columns
            for col in self.categorical_cols:
                if col in X.columns:
                    self.logger.info(f"Transforming categorical column: {col}")
                    le = self.encoders[col]
                    # Handle unseen categories
                    mask = ~X[col].astype(str).isin(le.classes_)
                    if mask.any():
                        unseen_count = mask.sum()
                        unseen_values = X.loc[mask, col].unique()
                        self.logger.warning(f"Found {unseen_count} rows with unseen values in column {col}: {unseen_values[:5]}{'...' if len(unseen_values) > 5 else ''}")
                        X.loc[mask, col] = 'unknown'
                    try:
                        X[col] = le.transform(X[col].astype(str))
                        self.logger.info(f"Column {col} transformed successfully")
                    except ValueError as ve:
                        self.logger.error(f"Error transforming column {col}: {str(ve)}")
                        # Fallback for completely new categories
                        X[col] = 0  # Assign a default value
                        self.logger.warning(f"Used fallback value 0 for column {col}")
                else:
                    self.logger.warning(f"Column {col} not found in input data")
                    
            self.logger.info(f"Feature transformation completed. Output shape: {X.shape}")
            return X
        except Exception as e:
            self.logger.error(f"Error in transform method: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

def predict_new_loans(output_path=None):
    """
    Make predictions on new loan data and save the results
    """
    if output_path is None:
        output_path = PREDICTION_DIR
        
    logger.info("Starting loan prediction process")
    try:
        # Create directories if they don't exist
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory confirmed: {output_path}")
        
        # Load model
        model_path = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load new data
        data_path = os.path.join(DATA_DIR, "sample_loan_data.csv")
        logger.info(f"Loading data from: {data_path}")
        new_data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {new_data.shape}")
        
        # Basic data validation
        logger.info("Performing basic data validation")
        missing_values = new_data.isnull().sum().sum()
        logger.info(f"Missing values in the dataset: {missing_values}")
        
        # Log data summary statistics
        logger.info("Data summary statistics:")
        for col in new_data.columns:
            if new_data[col].dtype in ['int64', 'float64']:
                logger.info(f"{col}: min={new_data[col].min()}, max={new_data[col].max()}, mean={new_data[col].mean():.2f}, median={new_data[col].median()}")
            else:
                logger.info(f"{col}: {len(new_data[col].unique())} unique values")
        
        # Create and fit preprocessor
        logger.info("Initializing feature preprocessor")
        preprocessor = FeaturePreprocessor()
        
        # Fit on data (excluding target if present)
        if 'loanRisk' in new_data.columns:
            logger.info("Target column 'loanRisk' found in data")
            fit_data = new_data.drop('loanRisk', axis=1)
        else:
            logger.info("Target column 'loanRisk' not found in data")
            fit_data = new_data.copy()
            
        logger.info("Fitting feature preprocessor")
        preprocessor.fit(fit_data)
        
        # Transform features
        logger.info("Transforming features")
        X_new = preprocessor.transform(new_data)
        logger.info(f"Features transformed successfully. Shape: {X_new.shape}")
        
        # Make predictions
        logger.info("Making predictions")
        start_time = datetime.now()
        predictions_proba = model.predict(X_new)  # Get probability predictions (n_samples × n_classes)
        end_time = datetime.now()
        prediction_time = (end_time - start_time).total_seconds()
        logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
        
        predictions_class = np.argmax(predictions_proba, axis=1)  # Get class predictions
        logger.info(f"Class distribution in predictions: {np.bincount(predictions_class)}")
        
        # Map class predictions to risk labels
        risk_labels = {0: 'High Risk', 1: 'Low Risk', 2: 'Moderate Risk'}
        logger.info(f"Risk label mapping: {risk_labels}")
        
        # Prepare results dataframe
        logger.info("Preparing results dataframe")
        results = new_data.copy()
        results['loanRisk_numeric'] = predictions_class
        results['loanRisk'] = [risk_labels.get(pred, 'Unknown') for pred in predictions_class]
        
        # Calculate risk distribution
        risk_distribution = results['loanRisk'].value_counts()
        logger.info(f"Risk distribution in predictions: {risk_distribution.to_dict()}")
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_path, f"predictions_{timestamp}.csv")
        logger.info(f"Saving predictions to: {output_file}")
        results.to_csv(output_file, index=False)
        logger.info(f"Predictions saved successfully")
        
        print(f"Predictions saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error in predict_new_loans: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        logger.info("=== Starting Loan Prediction Pipeline ===")
        output_file = predict_new_loans()
        logger.info(f"=== Loan Prediction Pipeline Completed Successfully ===")
        logger.info(f"Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())