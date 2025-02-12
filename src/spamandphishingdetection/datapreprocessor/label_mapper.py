from sklearn.base import BaseEstimator, TransformerMixin

class LabelMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'label' in X.columns:
            X['label'] = X['label'].map(self.mapping)
        return X
