from sklearn.base import BaseEstimator, TransformerMixin
import logging
import pandas as pd


class LabelMapper(BaseEstimator, TransformerMixin):
    """
    Transformer that maps labels based on a provided mapping dictionary.

    Parameters:
    mapping (dict): A dictionary mapping original labels to new labels.
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            logging.error("LabelMapper.fit received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            logging.error("LabelMapper.transform received non-DataFrame input")
            raise ValueError("Input must be a pandas DataFrame")
        X = X.copy()
        if 'label' in X.columns:
            X['label'] = X['label'].map(self.mapping)
        else:
            logging.warning(
                "LabelMapper.transform: 'label' column not found in DataFrame")
        return X
