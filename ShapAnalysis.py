import joblib
import shap
import numpy as np
import pandas as pd
import os
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt  # Plotting library



logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)



base_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = [os.path.join(base_dir, 'Models & Parameters', f'ensemble_model_fold_{i+1}.pkl') for i in range(3)]
X_test_paths = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_test_data.npz') for i in range(3)]
y_test_paths = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_test_labels.pkl') for i in range(3)]
features_path = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_train_data.npz') for i in range(3)]
features_label = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_train_labels.npz') for i in range(3)]
preprocessor_path = [os.path.join(base_dir, 'Feature Extraction', f'fold_{i+1}_preprocessor.pkl') for i in range(3)]


# Define column names
categorical_columns = ['sender', 'receiver', 'has_ip_address']
numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls']


# Recreate the ColumnTransformer excluding 'cleaned_text' and 'label'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), categorical_columns),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_columns)
    ],
    remainder='passthrough'  # Keep other columns unchanged
)
column_names = categorical_columns + numerical_columns 


# Load models, data, and feature names for each fold
for fold in range(3):
    logging.info(f'Processing fold {fold+1}')
    
    # Load the model for this fold
    start_time = time.perf_counter()
    model = joblib.load(model_paths[fold])
    logging.info(f'Loaded model from {model_paths[fold]}')
    logging.info(f'Time taken to load model: {time.perf_counter() - start_time:.2f} seconds\n')

    # Load the test data
    start_time = time.perf_counter()
    X_test_combined = np.load(X_test_paths[fold])['data']
    logging.info(f'Loaded test data from {X_test_paths[fold]}')
    logging.info(f'Time taken to load test data: {time.perf_counter() - start_time:.2f} seconds\n')

    # Load the training data to extract feature names
    start_time = time.perf_counter()
    X_train_combined = np.load(features_path[fold])['data']
    logging.info(f'Loaded train data from {features_path[fold]}')
    logging.info(f'Time taken to load train data: {time.perf_counter() - start_time:.2f} seconds\n')

    # Load the preprocessor
    start_time = time.perf_counter()
    preprocessor = joblib.load(preprocessor_path[fold])
    logging.info(f'Loaded preprocessor')
    logging.info(f'Time taken to load preprocessor: {time.perf_counter() - start_time:.2f} seconds\n')

    # Extract feature names
    start_time = time.perf_counter()
    feature_names_non_text = preprocessor.get_feature_names_out()  # Original non-text feature names
    feature_names_text = [f"bert_feature_{i}" for i in range(X_train_combined.shape[1] - len(feature_names_non_text))]
    feature_names = np.concatenate([feature_names_non_text, feature_names_text])
    logging.info(f'Extracted feature names')
    logging.info(f'Time taken to extract feature names: {time.perf_counter() - start_time:.2f} seconds\n')

    # Define a function to predict using the model
    def predict_fn(X):
        return model.predict_proba(X)

    # Reduce the number of background samples using ksample
    start_time = time.perf_counter()
    logging.info('Reducing number of background samples using ksample...')
    sample_size = min(1000, X_train_combined.shape[0])  # Sample up to 1000 data points
    X_train_sample = shuffle(X_train_combined, n_samples=sample_size, random_state=42)
    background_samples = shap.sample(X_train_sample, 100)  # Perform ksample with reduced sample size
    logging.info('Completed ksample sampling.')
    logging.info(f'Time taken for ksample sampling: {time.perf_counter() - start_time:.2f} seconds\n')

    # Initialize SHAP explainer based on model type
    start_time = time.perf_counter()
    if hasattr(model, 'named_estimators_'):
        model_rf = model.named_estimators_['rf']
        logging.info('Initializing SHAP TreeExplainer...')
        explainer = shap.TreeExplainer(model_rf, background_samples)
    else:
        logging.info('Initializing SHAP KernelExplainer...')
        explainer = shap.KernelExplainer(predict_fn, background_samples)
    logging.info('Initialized SHAP Explainer.')
    logging.info(f'Time taken to initialize SHAP Explainer: {time.perf_counter() - start_time:.2f} seconds\n')

    # Calculate SHAP values
    start_time = time.perf_counter()
    logging.info('Calculating SHAP values...')
    X_test_subset = X_test_combined[:2000]  # Use a subset of the test data
    shap_values = explainer.shap_values(X_test_subset, check_additivity=False)
    logging.info('Completed SHAP value calculation.')
    logging.info(f'Time taken to calculate SHAP values: {time.perf_counter() - start_time:.2f} seconds\n')

    # Generate SHAP summary plot
    logging.info('Generating SHAP summary plot...')
    plt.figure(figsize=(12, 8))  # Adjust the figure size for better visibility
    shap.summary_plot(shap_values, X_test_subset, feature_names=feature_names, max_display=20, plot_type="bar")  # Limit the number of features displayed
    plot_path = os.path.join(base_dir, f'shap_summary_plot_fold_{fold+1}.png')
    plt.savefig(plot_path)
    logging.info(f'SHAP summary plot saved to {plot_path}.')
    plt.close()  # Close the plot to free up memory

    # If it's the first fold, you might want to stop early for testing
    if fold == 0:
        logging.info('Ending after the first fold for testing.')
        break  # Exit the loop after processing the first fold