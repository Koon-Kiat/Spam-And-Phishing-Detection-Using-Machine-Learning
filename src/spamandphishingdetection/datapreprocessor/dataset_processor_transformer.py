import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Import existing modules
from src.spamandphishingdetection.datapreprocessor.dataset_processor import DatasetProcessor
from src.spamandphishingdetection.datapreprocessor.label_mapper import LabelMapper
from src.spamandphishingdetection.validator.log_label_percentage import log_label_percentages

class DatasetProcessorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that processes a dataset by applying DatasetProcessor. Expects a pandas DataFrame with the required column specified by column_name.
    """

    def __init__(self, column_name, dataset_name, save_path):
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Validate input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            logging.error(
                "DatasetProcessorTransformer.transform received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")
        # Validate required column
        if self.column_name not in X.columns:
            logging.error(
                f"DatasetProcessorTransformer.transform: Missing required column '{self.column_name}' in DataFrame")
            raise ValueError(
                f"Input DataFrame must contain column '{self.column_name}'")

        processor = DatasetProcessor(
            X, self.column_name, self.dataset_name, self.save_path)
        processed_df = processor.process()
        return processed_df


class LabelLoggingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that logs label percentages. Expects a DataFrame with a 'label' column.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def fit(self, X, y=None):
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
        return X


def build_spamassassin_pipeline(save_path):
    spam_mapping = {1: 0, 0: 1}
    pipeline = Pipeline([
        ('label_mapping', LabelMapper(spam_mapping)),
        ('dataset_processing', DatasetProcessorTransformer(
            'text', 'spam_assassin', save_path)),
        ('label_logging', LabelLoggingTransformer('SpamAssassin'))
    ])
    return pipeline


def build_ceas_pipeline(save_path):
    pipeline = Pipeline([
        ('dataset_processing', DatasetProcessorTransformer(
            'body', 'ceas_08', save_path)),
        ('label_logging', LabelLoggingTransformer('CEAS_08'))
    ])
    return pipeline
