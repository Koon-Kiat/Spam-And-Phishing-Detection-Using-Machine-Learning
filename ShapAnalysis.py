import joblib
import numpy as np
import pandas as pd
import shap
import os
import logging
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tqdm import tqdm  # Import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to your saved models, parameters, and features
model_paths = [os.path.join(base_dir, 'Models & Parameters', f'ensemble_model_fold_{i+1}.pkl') for i in range(3)]
param_paths = [os.path.join(base_dir, 'Models & Parameters', f'best_params_fold_{i+1}.json') for i in range(3)]
feature_paths = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_test_data.npz') for i in range(3)]
feature_label_paths = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_test_labels.pkl') for i in range(3)]
X_test_paths = [os.path.join(base_dir, 'Data Splitting', f'X_test_fold{i+1}.csv') for i in range(3)]
y_test_paths = [os.path.join(base_dir, 'Data Splitting', f'y_test_fold{i+1}.csv') for i in range(3)]

def load_or_save_model(model_path, action='load'):
    if action == 'load':
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
        else:
            logging.info(f"No saved model found at {model_path}. Proceeding to train a new model.")
            return None
    elif action == 'save':
        raise NotImplementedError("Save action is not implemented in this function.")

# Use tqdm for progress tracking
for i, (model_path, param_path, feature_path, feature_label_path, X_test_path, y_test_path) in enumerate(
        zip(model_paths, param_paths, feature_paths, feature_label_paths, X_test_paths, y_test_paths), start=1):
    
    logging.info(f"Processing fold {i}...")

    # Load the saved model using joblib
    model = load_or_save_model(model_path, action='load')
    
    if model is None:
        logging.error(f"Model could not be loaded from {model_path}")
        continue

    logging.info("Model loaded successfully.")

    # Load the saved features from .npz file
    try:
        with np.load(feature_path) as data:
            X_test_features = data['data']  # Use the 'data' key as it matches your saving method
    except KeyError as e:
        logging.error(f"KeyError while loading features from {feature_path}: {e}")
        continue

    logging.info("Features loaded successfully.")

    # Load the saved labels from .pkl file
    try:
        with open(feature_label_path, 'rb') as f:
            y_test_labels = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading labels from {feature_label_path}: {e}")
        continue

    logging.info("Labels loaded successfully.")

    # Load X_test and y_test data from CSV files (if needed)
    try:
        X_test_csv = pd.read_csv(X_test_path)
        y_test_csv = pd.read_csv(y_test_path)
    except Exception as e:
        logging.error(f"Error loading X_test or y_test from CSV files: {e}")
        continue

    logging.info("CSV data loaded successfully.")

    # Determine model type for SHAP explainer
    try:
        if isinstance(model, VotingClassifier):
            # Extract base models correctly
            for name, base_model in model.named_estimators_.items():
                if isinstance(base_model, RandomForestClassifier):
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_test_features)
                    np.save(os.path.join(base_dir, f'shap_values_{name}_fold_{i}.npy'), shap_values)
                    shap.summary_plot(shap_values, X_test_features)
                elif isinstance(base_model, LogisticRegression):
                    explainer = shap.KernelExplainer(base_model.predict_proba, X_test_features)
                    shap_values = explainer.shap_values(X_test_features)
                    np.save(os.path.join(base_dir, f'shap_values_{name}_fold_{i}.npy'), shap_values)
                    shap.summary_plot(shap_values, X_test_features)
        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_features)
            np.save(os.path.join(base_dir, f'shap_values_fold_{i}.npy'), shap_values)
            shap.summary_plot(shap_values, X_test_features)
        elif isinstance(model, LogisticRegression):
            explainer = shap.KernelExplainer(model.predict_proba, X_test_features)
            shap_values = explainer.shap_values(X_test_features)
            np.save(os.path.join(base_dir, f'shap_values_fold_{i}.npy'), shap_values)
            shap.summary_plot(shap_values, X_test_features)
        else:
            logging.warning(f"SHAP explanation not supported for model type: {type(model)}")
    except Exception as e:
        logging.error(f"Error during SHAP explanation: {e}")

    logging.info(f"Fold {i} processing completed.")

logging.info("All folds processed.")
