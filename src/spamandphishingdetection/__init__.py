"""
spamandphishingdetection: A package for detecting spam and phishing emails using machine learning.

This package provides various modules for data processing, feature engineering, model training,
and evaluation in the context of spam and phishing detection.
"""

import os

# Import components from subpackages
# Data cleaning and preprocessing
from .datacleaning.data_cleaning_pipeline_api import run_data_cleaning_pipeline
from .datacleaning.headers_api import run_headers_cleaning_pipeline
from .datapreprocessor.pipeline_builders import build_spamassassin_pipeline, build_ceas_pipeline
from .dataintegration.data_integrator import integrate_datasets

# Initialization and utilities
from .initializer.setup import initialize_environment
from .initializer.file_operations import load_config, check_config, get_file_paths, get_model_path, get_params_path

# Validation and verification
from .validator.log_label_percentage import log_label_percentages
from .validator.check_missing_values import check_missing_values
from .validator.verify_dataframe import verify_dataframe

# Feature engineering and data processing
from .featureengineering.email_header_extractor import feature_engineering, count_urls
from .noise_injection import generate_noisy_dataframe
from .data_splitting import stratified_k_fold_split
from .bert import BERTFeatureExtractor, BERTFeatureTransformer
from .rare_category_remover import RareCategoryRemover
from .pipeline import run_pipeline_or_load
from .learning_curve import plot_learning_curve

# Model training
from .modeltraining.base_model import model_training as base_model_training
from .modeltraining.main_model import model_training as main_model_training
from .modeltraining.base_model_optuna import model_training as base_model_training_optuna
from .modeltraining.xgb_lightgb_lg_model import model_training as xgb_lightgb_lg_model_training
from .modeltraining.xgb_rf_lg_model import model_training as xgb_rf_lg_model_training
from .modeltraining.xgb_ada_lg_model import model_training as xgb_ada_lg_model_training
from .modeltraining.xgb_knn_lg_model import model_training as xgb_knn_lg_model_training

__version__ = "0.1.0"
