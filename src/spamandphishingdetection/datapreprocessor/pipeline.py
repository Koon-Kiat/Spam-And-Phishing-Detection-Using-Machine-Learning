import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Import existing modules
from src.spamandphishingdetection.datapreprocessor.dataset_processor import DatasetProcessor
from src.spamandphishingdetection.datapreprocessor.label_processing import log_label_percentages
from src.spamandphishingdetection.datapreprocessor.label_mapper import LabelMapper


class DatasetProcessorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, dataset_name, save_path):
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processor = DatasetProcessor(
            X, self.column_name, self.dataset_name, self.save_path)
        processed_df = processor.process_dataset()
        return processed_df


class LabelLoggingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def fit(self, X, y=None):
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
