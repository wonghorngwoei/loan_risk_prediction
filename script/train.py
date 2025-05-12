import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
from datetime import datetime
import json
import os
import logging
import traceback
import sys
import time

# === Directory Structure Setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Logger Setup ===
# Generate a unique run ID for this execution
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOG_DIR, f'run_{RUN_ID}')
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# Create a symlink or copy for the "latest" run
LATEST_LOG_DIR = os.path.join(LOG_DIR, 'latest_training')
if os.path.exists(LATEST_LOG_DIR):
    if os.path.islink(LATEST_LOG_DIR):
        os.unlink(LATEST_LOG_DIR)
    else:
        import shutil
        shutil.rmtree(LATEST_LOG_DIR)

try:
    # Try to create symlink first (works on Unix/Linux/MacOS)
    os.symlink(RUN_LOG_DIR, LATEST_LOG_DIR)
except (OSError, AttributeError):
    # Fall back to creating a copy of the directory (for Windows)
    os.makedirs(LATEST_LOG_DIR, exist_ok=True)

def setup_logger(name, log_file=None):
    """Set up a logger with file and console handlers"""
    if log_file is None:
        log_file = os.path.join(RUN_LOG_DIR, f'{name}.log')
    
    # Create logger
    logger = logging.getLogger(f"{name}_{RUN_ID}")  # Make logger name unique per run
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any (to avoid duplicate logging)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Apply formatters to handlers
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also save a copy to the latest log directory
    latest_file_handler = logging.FileHandler(os.path.join(LATEST_LOG_DIR, f'{name}.log'))
    latest_file_handler.setFormatter(detailed_formatter)
    logger.addHandler(latest_file_handler)
    
    return logger

# Initialize main logger
logger = setup_logger('model_training')

def prepare_data(loan_data):
    """Prepare data from existing loan_data DataFrame"""
    logger.info("Starting data preparation")
    logger.info(f"Initial data shape: {loan_data.shape}")
    
    # Check for missing values in target
    missing_target = loan_data['loanRisk'].isna().sum()
    logger.info(f"Missing values in loanRisk target: {missing_target}")
    
    # Log overall missing values
    missing_counts = loan_data.isna().sum()
    total_missing = missing_counts.sum()
    logger.info(f"Total missing values in dataset: {total_missing}")
    
    # Log columns with high missing values
    high_missing = missing_counts[missing_counts > 0]
    if not high_missing.empty:
        logger.info(f"Columns with missing values:\n{high_missing}")
    
    # Drop rows with missing target
    loan_data = loan_data.dropna(subset=['loanRisk'])
    logger.info(f"Data shape after dropping missing targets: {loan_data.shape}")
    
    # Check value distribution in target
    target_distribution = loan_data['loanRisk'].value_counts()
    logger.info(f"Target class distribution:\n{target_distribution}")
    
    # Extract features and target
    y = loan_data['loanRisk']
    X = loan_data.drop(['loanRisk', 'clearfraudscore'], axis=1)
    logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
    
    # Log feature types
    dtypes_count = X.dtypes.value_counts()
    logger.info(f"Feature data types:\n{dtypes_count}")
    
    return X, y

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.logger = setup_logger('feature_preprocessor')
        self.logger.info(f"Feature preprocessor initialized for run {RUN_ID}")

    def fit_transform(self, X):
        self.logger.info("Starting feature preprocessing fit_transform")
        try:
            X = X.copy()  # Don't modify the original
            start_time = time.time()
            
            # Process numeric columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            self.logger.info(f"Identified {len(numeric_cols)} numeric columns")
            
            if not numeric_cols.empty:
                self.logger.info("Applying StandardScaler to numeric columns")
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
                self.logger.info(f"StandardScaler mean values: {self.scaler.mean_[:5]}...")
                self.logger.info(f"StandardScaler scale values: {self.scaler.scale_[:5]}...")

            # Process categorical columns
            categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns
            self.logger.info(f"Identified {len(categorical_cols)} categorical columns")
            
            for col in categorical_cols:
                unique_values = X[col].astype(str).nunique()
                self.logger.info(f"Encoding column '{col}' with {unique_values} unique values")
                
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                self.logger.info(f"LabelEncoder classes for '{col}': {le.classes_[:5]}...")
            
            processing_time = time.time() - start_time
            self.logger.info(f"Feature preprocessing completed in {processing_time:.2f} seconds")
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error in fit_transform: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def transform(self, X):
        self.logger.info("Starting feature preprocessing transform")
        try:
            X = X.copy()  # Don't modify the original
            
            # Process numeric columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                self.logger.info(f"Transforming {len(numeric_cols)} numeric columns")
                X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            
            # Process categorical columns
            for col, le in self.encoders.items():
                if col in X.columns:
                    self.logger.info(f"Transforming categorical column: {col}")
                    
                    # Handle unseen categories
                    unique_before = X[col].nunique()
                    mask = ~X[col].astype(str).isin(le.classes_)
                    
                    if mask.any():
                        unseen_count = mask.sum()
                        unseen_values = X.loc[mask, col].unique()
                        self.logger.warning(f"Found {unseen_count} rows with unseen categories in '{col}': {unseen_values[:5]}{'...' if len(unseen_values) > 5 else ''}")
                        # Handle unseen values with a placeholder
                        X.loc[mask, col] = le.classes_[0]  # Use the first class as a placeholder
                    
                    X[col] = le.transform(X[col].astype(str))
                    unique_after = X[col].nunique()
                    self.logger.info(f"Column '{col}' transformed. Unique values before: {unique_before}, after: {unique_after}")
                else:
                    self.logger.warning(f"Column '{col}' expected but not found in transform data")
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error in transform: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

def train_model(X_train, y_train, class_count):
    """Train LightGBM model with logging"""
    logger.info("Starting model training")
    
    # Log model hyperparameters
    model_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_class': class_count,
        'learning_rate': 0.08,           
        'num_leaves': 50,                              
        'min_data_in_leaf': 100,         
        'feature_fraction': 0.7,                     
        'verbose': -1,
        'is_unbalance': True
    }
    
    logger.info(f"Model hyperparameters: {json.dumps(model_params, indent=2)}")
    
    # Create and train model
    try:
        start_time = time.time()
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = model.feature_name_
            importances = model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            top_features = [(feature_names[i], importances[i]) for i in indices[:20]]
            
            logger.info("Top 20 feature importances:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model and log results"""
    logger.info("Starting model evaluation")
    
    try:
        # Make predictions
        start_time = time.time()
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # Multiclass AUC calculation
        auc_score = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class='ovr',
            average='weighted'
        )
        
        logger.info(f"Weighted AUC-ROC: {auc_score:.4f}")
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        text_report = classification_report(y_test, y_pred)
        
        # Log overall metrics
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        logger.info(f"Macro avg precision: {report['macro avg']['precision']:.4f}")
        logger.info(f"Macro avg recall: {report['macro avg']['recall']:.4f}")
        logger.info(f"Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix:")
        for row in cm:
            logger.info(f"  {row}")
        
        # Calculate and log class-specific metrics
        logger.info("Class-specific metrics:")
        for class_label in sorted(report.keys()):
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[class_label]
                logger.info(f"  Class {class_label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")
        
        return {
            "auc": auc_score,
            "classification_report": report,
            "text_report": text_report,
            "confusion_matrix": cm.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_artifacts(model, metrics):
    """Save model, metrics"""
    logger.info("Saving model artifacts")
    
    # Use the global run ID for versioning to keep everything aligned
    version = RUN_ID
    
    try:
        # Save model
        model_path = os.path.join(MODEL_DIR, f"lightgbm_v{version}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save a copy as the latest model
        latest_model_path = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
        joblib.dump(model, latest_model_path)
        logger.info(f"Latest model copy saved to: {latest_model_path}")
        
        # # Save metrics
        # metrics_path = os.path.join(METRICS_DIR, f"metrics_v{version}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        
        # Add run info to metrics
        metrics["run_id"] = RUN_ID
        metrics["timestamp"] = datetime.now().isoformat()
        
        
        # Also save metrics to the run log directory for completeness
        run_metrics_path = os.path.join(RUN_LOG_DIR, "metrics.json")
        with open(run_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Run-specific metrics saved to: {run_metrics_path}")
        
        return version
        
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logger.info(f"=== STARTING LOAN RISK MODEL TRAINING PIPELINE (RUN ID: {RUN_ID}) ===")
    logger.info(f"Logs for this run will be saved to: {RUN_LOG_DIR}")
    pipeline_start_time = time.time()
    
    try:
        # Log system info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"LightGBM version: {lgb.__version__}")
        
        # Load data
        data_path = os.path.join(DATA_DIR, "final_loan_data.csv")
        logger.info(f"Loading data from: {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
        except FileNotFoundError:
            alternate_path = "notebook/final_loan_data.csv"
            logger.warning(f"Data not found at {data_path}, trying alternate path: {alternate_path}")
            data = pd.read_csv(alternate_path)
            logger.info(f"Data loaded from alternate path. Shape: {data.shape}")
        
        # Prepare data
        X, y = prepare_data(data)
        
        # Preprocess features
        logger.info("Initializing feature preprocessor")
        preprocessor = FeaturePreprocessor()
        
        logger.info("Applying feature preprocessing")
        X_processed = preprocessor.fit_transform(X)
        
        # Train/Test Split
        logger.info("Performing train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Check class distribution
        train_distribution = pd.Series(y_train).value_counts()
        logger.info(f"Original class distribution:\n{train_distribution}")
        
        # Apply SMOTE for class balancing
        class_count = len(np.unique(y_train))
        logger.info(f"Number of classes: {class_count}")
        
        if class_count >= 2:
            logger.info("Applying SMOTE oversampling")
            try:
                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                
                balanced_distribution = pd.Series(y_train_smote).value_counts()
                logger.info(f"Post-SMOTE distribution:\n{balanced_distribution}")
                
                oversampling_ratio = len(y_train_smote) / len(y_train)
                logger.info(f"Oversampling ratio: {oversampling_ratio:.2f}x")
                
            except Exception as e:
                logger.error(f"SMOTE failed: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("Using original unbalanced data due to SMOTE failure")
                X_train_smote, y_train_smote = X_train, y_train
        else:
            logger.warning("Insufficient classes for SMOTE. Using original data.")
            X_train_smote, y_train_smote = X_train, y_train
        
        # Train model
        model = train_model(X_train_smote, y_train_smote, class_count=y.nunique())
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save artifacts
        version = save_artifacts(model, metrics)
        
        # Log overall pipeline metrics
        pipeline_time = time.time() - pipeline_start_time
        logger.info(f"Total pipeline execution time: {pipeline_time:.2f} seconds")
        logger.info(f"Model version: {version}")
        logger.info(f"AUC score: {metrics['auc']:.4f}")
        logger.info("=== MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")
        
        # Print final report to console
        print("\nClassification Report:")
        print(metrics["text_report"])
        print(f"\nWeighted AUC-ROC: {metrics['auc']:.4f}")
        print(f"\nModel saved to: {os.path.join(MODEL_DIR, f'lightgbm_v{version}.pkl')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=== MODEL TRAINING PIPELINE FAILED ===")
        sys.exit(1)