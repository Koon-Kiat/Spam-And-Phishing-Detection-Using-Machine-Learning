"""
spamandphishingdetection: A package for detecting spam and phishing emails using machine learning.

This package provides various modules for data processing, feature engineering, model training,
and evaluation in the context of spam and phishing detection.
"""

import os

# Importing specific classes or functions to make them available at the package level
from .setup import initialize_environment
from .file_operations import load_config, get_file_paths, get_model_path, get_params_path
from .dataset_processor import DatasetProcessor
from .label_processing import log_label_percentages
from .missing_values import check_missing_values
from .feature_engineering import feature_engineering, count_urls
from .data_cleaning_headers import load_or_save_emails, process_and_save_emails
from .data_integration import (
    merge_dataframes,
    verify_merged_dataframe,
    combine_dataframes,
    verify_combined_dataframe,
    save_combined_dataframe
)
from .data_cleaning import (
    load_or_clean_data,
    data_cleaning,
    save_dataframe_to_csv,
    combine_columns_for_cleaning
)
from .noise_injection import generate_noisy_dataframe
from .data_splitting import stratified_k_fold_split
from .bert import BERTFeatureExtractor, BERTFeatureTransformer
from .rare_category_remover import RareCategoryRemover
from .pipeline import run_pipeline_or_load
from .learning_curve import plot_learning_curve

from .modeltraining.base_model import model_training as base_model_training
from .modeltraining.main_model import model_training as main_model_training
from .modeltraining.base_model_optuna import model_training as base_model_training_optuna
from .modeltraining.xgb_lightgb_lg_model import model_training as xgb_lightgb_lg_model_training
from .modeltraining.xgb_rf_lg_model import model_training as xgb_rf_lg_model_training
from .modeltraining.xgb_ada_lg_model import model_training as xgb_ada_lg_model_training
from .modeltraining.xgb_knn_lg_model import model_training as xgb_knn_lg_model_training

__version__ = "0.1.0"
