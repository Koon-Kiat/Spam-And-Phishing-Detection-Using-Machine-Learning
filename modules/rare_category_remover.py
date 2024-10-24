import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryRemover(BaseEstimator, TransformerMixin):
    """
    A custom transformer to remove rare categories from categorical features.

    Parameters
    ----------
    threshold : float, optional
        The frequency threshold below which categories are considered rare. Default is 0.05.

    Attributes
    ----------
    replacements_ : dict
        A dictionary mapping each column to another dictionary of rare categories and their replacements.
    all_categories_ : dict
        A dictionary storing all categories for each column before transformation.
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.replacements_ = {}
        self.all_categories_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer to the data by identifying rare categories.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data with categorical features.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Ensure columns are categorical
        for column in X.columns:
            X[column] = X[column].astype('category')
            frequency = X[column].value_counts(normalize=True)
            rare_categories = frequency[frequency < self.threshold].index
            self.replacements_[column] = {
                cat: 'Other' for cat in rare_categories}
            self.all_categories_[column] = X[column].unique()
        return self

    def transform(self, X):
        """
        Replace rare categories in the data using the mappings from the fit step.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data with categorical features.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The transformed data with rare categories replaced by 'Other'.
        """
        X_transformed = X.copy()
        for column in X.columns:
            X_transformed[column] = X[column].astype(
                'category')  # Ensure category dtype
            X_transformed[column] = X_transformed[column].replace(
                self.replacements_[column])

            # Ensure consistency by adding back any missing categories as 'Other'
            missing_categories = set(self.all_categories_[
                                     column]) - set(X_transformed[column].unique())
            if missing_categories:
                X_transformed[column] = X_transformed[column].cat.add_categories(
                    list(missing_categories))

        return X_transformed
