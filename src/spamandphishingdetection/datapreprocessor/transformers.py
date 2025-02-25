"""
Data Preprocessing Transformers
------------------------------
This module provides scikit-learn compatible transformers for data preprocessing tasks.
"""

import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessor import DataPreprocessor
from src.spamandphishingdetection.validator.log_label_percentage import log_label_percentages


class LabelMapper(BaseEstimator, TransformerMixin):
    """Transformer that maps categorical labels based on a provided mapping dictionary.

    This transformer follows the Single Responsibility Principle by focusing solely
    on mapping labels from one set of values to another.

    Parameters:
        mapping (dict): Dictionary mapping original labels to new labels.
    """

    def __init__(self, mapping):
        """Initialize the LabelMapper with a mapping dictionary."""
        self.mapping = mapping

    def fit(self, X, y=None):
        """Validate input data format.

        Parameters:
            X (pd.DataFrame): Input data
            y: Ignored, exists for scikit-learn compatibility

        Returns:
            self: Reference to self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            logging.error("LabelMapper.fit received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")
        return self

    def transform(self, X):
        """Apply the mapping to labels in the DataFrame.

        Parameters:
            X (pd.DataFrame): Input DataFrame with a 'label' column

        Returns:
            pd.DataFrame: DataFrame with mapped labels
        """
        if not isinstance(X, pd.DataFrame):
            logging.error("LabelMapper.transform received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")

        X_transformed = X.copy()

        if 'label' in X_transformed.columns:
            X_transformed['label'] = X_transformed['label'].map(self.mapping)
            missing_labels = X_transformed['label'].isna().sum()
            if missing_labels > 0:
                logging.warning(
                    f"LabelMapper found {missing_labels} values with no mapping")
        else:
            logging.warning(
                "LabelMapper.transform: 'label' column not found in DataFrame")

        return X_transformed


class DatasetPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Transformer that preprocesses a dataset using the DataPreprocessor.

    This transformer adapts the DataPreprocessor class to be used in scikit-learn pipelines,
    providing a standardized interface for dataset preprocessing operations.

    Parameters:
        column_name (str): Column used to identify duplicates.
        dataset_name (str): Identifier for logging.
        save_path (str): File path to save the processed DataFrame.
    """

    def __init__(self, column_name, dataset_name, save_path):
        """Initialize the transformer with preprocessing parameters."""
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path

    def fit(self, X, y=None):
        """Fit method (no-op) for scikit-learn compatibility.

        Parameters:
            X: Input features (ignored)
            y: Target values (ignored)

        Returns:
            self: Reference to self for method chaining
        """
        return self

    def transform(self, X):
        """Transform the dataset using the DataPreprocessor.

        Parameters:
            X (pd.DataFrame): Input DataFrame to preprocess

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Validate input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            logging.error(
                "DatasetPreprocessorTransformer.transform received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")

        # Validate required column
        if self.column_name not in X.columns:
            logging.error(
                f"DatasetPreprocessorTransformer.transform: Missing required column '{self.column_name}' in DataFrame")
            raise ValueError(
                f"Input DataFrame must contain column '{self.column_name}'")

        # Process the dataset
        preprocessor = DataPreprocessor(
            X, self.column_name, self.dataset_name, self.save_path)
        processed_df = preprocessor.process()

        return processed_df


class LabelLoggingTransformer(BaseEstimator, TransformerMixin):
    """Transformer that logs label distribution statistics in datasets.

    This transformer is responsible for monitoring and logging the distribution of labels,
    following the Single Responsibility Principle.

    Parameters:
        dataset_name (str): Identifier for logging.
    """

    def __init__(self, dataset_name):
        """Initialize the transformer with a dataset name for logging."""
        self.dataset_name = dataset_name

    def fit(self, X, y=None):
        """Log label percentages during fit.

        Parameters:
            X (pd.DataFrame): Input DataFrame with a 'label' column
            y: Ignored, exists for scikit-learn compatibility

        Returns:
            self: Reference to self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            logging.error(
                "LabelLoggingTransformer.fit received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")

        if 'label' not in X.columns:
            logging.warning(
                "LabelLoggingTransformer.fit: 'label' column not found in DataFrame")
        else:
            # Log label percentages during fit
            log_label_percentages(X, self.dataset_name)

        return self

    def transform(self, X):
        """Pass-through transform method.

        Parameters:
            X (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Unchanged input DataFrame
        """
        return X
