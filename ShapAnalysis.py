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
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ",
    level=logging.INFO,
)


class RareCategoryRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.replacements_ = {}

    def fit(self, X, y=None):
        logging.info(f"Removing rare categories with threshold: {self.threshold}")
        for column in X.columns:
            frequency = X[column].value_counts(normalize=True)
            rare_categories = frequency[frequency < self.threshold].index
            self.replacements_[column] = {cat: "Other" for cat in rare_categories}

        return self

    def transform(self, X):
        for column, replacements in self.replacements_.items():
            X.loc[:, column] = X[column].replace(replacements)
        assert (
            X.shape[0] == X.shape[0]
        ), "Row count changed during rare category removal."

        return X

    def get_feature_names_out(self, input_features=None):
        return (
            input_features
            if input_features is not None
            else list(self.replacements_.keys())
        )


# Define paths to models, data, and feature names
base_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = [
    os.path.join(base_dir, "Models & Parameters", f"ensemble_model_fold_{i+1}.pkl")
    for i in range(3)
]
X_test_paths = [
    os.path.join(base_dir, "Feature Extraction", f"fold_{i+1}_test_data.npz")
    for i in range(3)
]
y_test_paths = [
    os.path.join(base_dir, "Feature Extraction", f"fold_{i+1}_test_labels.pkl")
    for i in range(3)
]
features_path = [
    os.path.join(base_dir, "Feature Extraction", f"fold_{i+1}_train_data.npz")
    for i in range(3)
]
features_label = [
    os.path.join(base_dir, "Feature Extraction", f"fold_{i+1}_train_labels.npz")
    for i in range(3)
]
preprocessor_path = [
    os.path.join(base_dir, "Feature Extraction", f"fold_{i+1}_preprocessor.pkl")
    for i in range(3)
]


# Define column names
categorical_columns = ["sender", "receiver"]
numerical_columns = [
    "https_count",
    "http_count",
    "blacklisted_keywords_count",
    "urls",
    "short_urls",
    "has_ip_address",
]


# Recreate the ColumnTransformer excluding 'cleaned_text' and 'label'
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                [
                    (
                        "rare_cat_remover",
                        RareCategoryRemover(threshold=0.05),
                    ),  # Remove rare categories
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Fill missing categorical values
                    (
                        "encoder",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ),
                ]
            ),
            categorical_columns,
        ),
        (
            "num",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="mean"),
                    ),  # Fill missing numerical values
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_columns,
        ),
    ],
    remainder="passthrough",  # Keep other columns unchanged, like 'cleaned_text' and 'label'
)
column_names = categorical_columns + numerical_columns


# Load models, data, and feature names for each fold
for fold in range(3):
    logging.info(f"Processing fold {fold+1}")

    # Load the model for this fold
    start_time = time.perf_counter()
    model = joblib.load(model_paths[fold])
    logging.info(f"Loaded model from {model_paths[fold]}")
    logging.info(
        f"Time taken to load model: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Load the test data
    start_time = time.perf_counter()
    X_test_combined = np.load(X_test_paths[fold])["data"]
    logging.info(f"Loaded test data from {X_test_paths[fold]}")
    logging.info(
        f"Time taken to load test data: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Load the training data to extract feature names
    start_time = time.perf_counter()
    X_train_combined = np.load(features_path[fold])["data"]
    logging.info(f"Loaded train data from {features_path[fold]}")
    logging.info(
        f"Time taken to load train data: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Load the preprocessor
    start_time = time.perf_counter()
    preprocessor = joblib.load(preprocessor_path[fold])
    logging.info(f"Loaded preprocessor")
    logging.info(
        f"Time taken to load preprocessor: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Extract feature names
    start_time = time.perf_counter()
    feature_names_non_text = (
        preprocessor.named_transformers_["cat"]
        .named_steps["rare_cat_remover"]
        .get_feature_names_out()
    )
    feature_names_text = [
        f"bert_feature_{i}"
        for i in range(X_train_combined.shape[1] - len(feature_names_non_text))
    ]
    feature_names = np.concatenate([feature_names_non_text, feature_names_text])
    logging.info(f"Extracted feature names")
    logging.info(
        f"Time taken to extract feature names: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Define a function to predict using the model
    def predict_fn(X):
        return model.predict_proba(X)

    # Reduce the number of background samples using ksample
    start_time = time.perf_counter()
    logging.info("Reducing number of background samples using ksample...")
    sample_size = min(1000, X_train_combined.shape[0])  # Sample up to 1000 data points
    X_train_sample = shuffle(X_train_combined, n_samples=sample_size, random_state=42)
    background_samples = shap.sample(
        X_train_sample, 100
    )  # Perform ksample with reduced sample size
    logging.info("Completed ksample sampling.")
    logging.info(
        f"Time taken for ksample sampling: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Initialize SHAP explainer based on model type
    start_time = time.perf_counter()
    if hasattr(model, "named_estimators_"):
        # Check for SVM model
        if "svm" in model.named_estimators_:
            model_svm = model.named_estimators_["svm"]
            logging.info("Initializing SHAP KernelExplainer for SVM...")
            explainer = shap.KernelExplainer(
                predict_fn, background_samples[:100]
            )  # Use a smaller subset of background samples
        # Check for XGBoost model
        elif "xgb" in model.named_estimators_:
            model_xgb = model.named_estimators_["xgb"]
            logging.info("Initializing SHAP TreeExplainer for XGBoost...")
            explainer = shap.TreeExplainer(
                model_xgb, background_samples[:100]
            )  # Use a smaller subset of background samples
        # Check for Meta model
        elif "meta" in model.named_estimators_:
            model_meta = model.named_estimators_["meta"]
            logging.info("Initializing SHAP KernelExplainer for Meta model...")
            explainer = shap.KernelExplainer(
                predict_fn, background_samples[:100]
            )  # Use a smaller subset of background samples
        else:
            logging.error(
                "No known model found in named_estimators_. Available keys are: %s",
                list(model.named_estimators_.keys()),
            )
            raise KeyError("No known model found in named_estimators_.")
    else:
        logging.info("Initializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn, background_samples[:100])

    logging.info("Initialized SHAP Explainer.")
    logging.info(
        f"Time taken to initialize SHAP Explainer: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Calculate SHAP values
    start_time = time.perf_counter()
    logging.info("Calculating SHAP values...")
    X_test_subset = X_test_combined[:500]  # Use a smaller subset of the test data
    shap_values = explainer.shap_values(X_test_subset, check_additivity=False)
    logging.info("Completed SHAP value calculation.")
    logging.info(
        f"Time taken to calculate SHAP values: {time.perf_counter() - start_time:.2f} seconds\n"
    )

    # Generate SHAP summary plot
    logging.info("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))  # Adjust the figure size for better visibility
    shap.summary_plot(
        shap_values,
        X_test_subset,
        feature_names=feature_names,
        max_display=20,
        plot_type="bar",
    )  # Limit the number of features displayed
    plot_path = os.path.join(base_dir, f"shap_summary_plot_fold_{fold+1}.png")
    plt.savefig(plot_path)
    logging.info(f"SHAP summary plot saved to {plot_path}.")
    plt.close()  # Close the plot to free up memory

    # If it's the first fold, you might want to stop early for testing
    if fold == 0:
        logging.info("Ending after the first fold for testing.")
        break  # Exit the loop after processing the first fold
