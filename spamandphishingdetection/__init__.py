"""
spamandphishingdetection: A package for detecting spam and phishing emails using machine learning.

This package provides various modules for data processing, feature engineering, model training,
and evaluation in the context of spam and phishing detection.
"""

# Importing specific classes or functions to make them available at the package level
from .file_operations import load_config, get_file_paths, get_model_path, get_params_path
from .dataset_processor import DatasetProcessor
from .label_processing import log_label_percentages
from .missing_values import check_missing_values
from .feature_engineering import feature_engineering
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
from .model_training import model_training
from .learning_curve import plot_learning_curve

__version__ = "0.1.0"
