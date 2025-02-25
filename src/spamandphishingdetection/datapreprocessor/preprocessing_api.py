"""
High-level API functions for the data preprocessing pipeline.

This module provides simplified functions to access the data preprocessing functionality,
making it easier to use without directly interacting with the underlying classes.
"""

import logging
import os
import pandas as pd

from src.spamandphishingdetection.datapreprocessor.preprocessor import DataPreprocessor
from src.spamandphishingdetection.datapreprocessor.pipeline_builders import (
    build_spamassassin_pipeline,
    build_ceas_pipeline
)


def preprocess_dataset(dataset, identifier_column, dataset_name, output_path):
    """
    Preprocess a dataset by removing unnamed columns, missing values, and duplicates.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input DataFrame to process.
    identifier_column : str
        Column name used to identify duplicates.
    dataset_name : str
        Identifier for logging purposes.
    output_path : str
        File path to save the processed DataFrame.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info(f"Starting data preprocessing for {dataset_name}")

    preprocessor = DataPreprocessor(
        dataset,
        identifier_column,
        dataset_name,
        output_path
    )

    try:
        processed_df = preprocessor.process()
        logging.info(f"Data preprocessing completed for {dataset_name}")
        return processed_df
    except Exception as e:
        logging.error(
            f"Error during data preprocessing for {dataset_name}: {e}")
        raise


def build_preprocessing_pipeline(dataset_type, save_path):
    """
    Build a preprocessing pipeline for a specific dataset type.

    Parameters
    ----------
    dataset_type : str
        The type of dataset ('spamassassin' or 'ceas').
    save_path : str
        Path to save the preprocessed dataset.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline for preprocessing the specified dataset type.
    """
    if dataset_type.lower() == 'spamassassin':
        return build_spamassassin_pipeline(save_path)
    elif dataset_type.lower() == 'ceas':
        return build_ceas_pipeline(save_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                         f"Supported types are 'spamassassin' and 'ceas'.")


def preprocess_with_pipeline(pipeline, dataframe):
    """
    Apply a preprocessing pipeline to a dataframe.

    Parameters
    ----------
    pipeline : Pipeline
        The scikit-learn pipeline to apply.
    dataframe : pandas.DataFrame
        The dataframe to process.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe.
    """
    try:
        processed_df = pipeline.fit_transform(dataframe)
        return processed_df
    except Exception as e:
        logging.error(f"Error applying preprocessing pipeline: {e}")
        raise
