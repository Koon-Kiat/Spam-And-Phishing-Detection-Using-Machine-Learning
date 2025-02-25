"""
Data Preprocessing Package
------------------------
This package provides tools and utilities for preprocessing email datasets.

The package includes transformers for label mapping, dataset preprocessing,
and scikit-learn compatible pipelines for standard preprocessing tasks.
"""

from src.spamandphishingdetection.datapreprocessor.preprocessing_api import (
    preprocess_dataset,
    build_preprocessing_pipeline,
    preprocess_with_pipeline
)

from src.spamandphishingdetection.datapreprocessor.pipeline_builders import (
    build_spamassassin_pipeline,
    build_ceas_pipeline
)

__all__ = [
    'preprocess_dataset',
    'build_preprocessing_pipeline',
    'preprocess_with_pipeline',
    'build_spamassassin_pipeline',
    'build_ceas_pipeline'
]
