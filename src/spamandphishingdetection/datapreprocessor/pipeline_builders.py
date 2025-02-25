"""
Pipeline Builders Module
----------------------
This module contains functions for building preprocessing pipelines for specific datasets.
"""

from sklearn.pipeline import Pipeline
from .transformers import (
    LabelMapper,
    DatasetPreprocessorTransformer,
    LabelLoggingTransformer
)


def build_spamassassin_pipeline(save_path):
    """Build a preprocessing pipeline for SpamAssassin dataset.

    This pipeline applies label mapping (where 1=ham, 0=spam gets mapped to 0=ham, 1=spam),
    dataset preprocessing, and label distribution logging.

    Parameters:
        save_path (str): Path to save the preprocessed dataset

    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing SpamAssassin data
    """
    # Map original labels (1=ham, 0=spam) to target format (0=ham, 1=spam)
    spam_mapping = {1: 0, 0: 1}

    pipeline = Pipeline([
        ('label_mapping', LabelMapper(spam_mapping)),
        ('dataset_preprocessing', DatasetPreprocessorTransformer(
            'text', 'SpamAssassin', save_path)),
        ('label_logging', LabelLoggingTransformer('SpamAssassin'))
    ])

    return pipeline


def build_ceas_pipeline(save_path):
    """Build a preprocessing pipeline for CEAS_08 dataset.

    This pipeline applies dataset preprocessing and label distribution logging.
    No label mapping is needed as CEAS labels already match the target format.

    Parameters:
        save_path (str): Path to save the preprocessed dataset

    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing CEAS_08 data
    """
    pipeline = Pipeline([
        ('dataset_preprocessing', DatasetPreprocessorTransformer(
            'body', 'CEAS_08', save_path)),
        ('label_logging', LabelLoggingTransformer('CEAS_08'))
    ])

    return pipeline
