"""
spamandphishingdetection: A package for detecting spam and phishing emails using machine learning.

This package provides various modules for data processing, feature engineering, model training,
and evaluation in the context of spam and phishing detection.
"""

import os

# Importing specific classes or functions to make them available at the package level
from .initializer.setup import initialize_environment
from .initializer.file_operations import load_config, check_config, get_file_paths, get_model_path, get_params_path
from .datapreprocessor.dataset_processor import DatasetProcessor
from .validator.log_label_percentage import log_label_percentages
from .datapreprocessor.label_mapper import LabelMapper
from .datapreprocessor.dataset_processor_transformer import DatasetProcessorTransformer, LabelLoggingTransformer, build_spamassassin_pipeline, build_ceas_pipeline
from .validator.check_missing_values import check_missing_values
from .featureengineering.email_header_extractor import feature_engineering, count_urls
from .datacleaning.headers_cleaner import get_email_headers, save_email_headers, get_merged_email_headers
from .dataintegration.data_integrator import integrate_datasets
from .validator.verify_dataframe import verify_dataframe

from .datacleaning.data_cleaning import (
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
