"""
Core Data Preprocessor Module
----------------------------
Contains the DataPreprocessor class for cleaning and preprocessing email datasets.
"""

import logging
import pandas as pd
import os


class DataPreprocessor:
    """Preprocess a dataset by removing unnamed columns, missing values, and duplicates.

    This class implements a sequence of data cleaning operations following the Single
    Responsibility Principle, with each method handling a specific cleaning task.

    Parameters:
        dataset (pd.DataFrame): Input DataFrame to process.
        identifier_column (str): Column name used to identify duplicates.
        dataset_name (str): Identifier for logging purposes.
        output_path (str): File path to save the processed DataFrame.
    """

    def __init__(self, dataset, identifier_column, dataset_name, output_path):
        """Initialize the DataPreprocessor with dataset and processing parameters."""
        self.dataset = dataset
        self.identifier_column = identifier_column
        self.dataset_name = dataset_name
        self.output_path = output_path

    def remove_unnamed_columns(self):
        """Remove 'Unnamed: X' columns commonly found in CSV imports.

        Returns:
            pd.DataFrame: DataFrame with unnamed columns removed.
        """
        unnamed_cols = [
            col for col in self.dataset.columns if 'Unnamed:' in col]
        if unnamed_cols:
            self.dataset = self.dataset.drop(columns=unnamed_cols)
            logging.info(
                f"{self.dataset_name}: Dropped {len(unnamed_cols)} unnamed column(s).")
        return self.dataset

    def remove_missing_values(self):
        """Log and drop rows containing missing values from the dataset.

        Returns:
            pd.DataFrame: DataFrame with missing values removed.
        """
        missing_counts = self.dataset.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logging.info(
                f"{self.dataset_name}: {total_missing} missing values found across {(missing_counts > 0).sum()} columns.")

            initial_count = self.dataset.shape[0]
            self.dataset = self.dataset.dropna()
            rows_removed = initial_count - self.dataset.shape[0]

            logging.info(
                f"{self.dataset_name}: Removed {rows_removed} rows with missing values. {self.dataset.shape[0]} rows remain.")
        else:
            logging.info(f"{self.dataset_name}: No missing values found.")

        return self.dataset

    def remove_duplicate_records(self):
        """Drop duplicate rows based on the identifier column.

        Returns:
            pd.DataFrame: DataFrame with duplicate rows removed.
        """
        if self.identifier_column not in self.dataset.columns:
            logging.warning(
                f"{self.dataset_name}: Identifier column '{self.identifier_column}' not found. Skipping duplicate removal.")
            return self.dataset

        initial_count = self.dataset.shape[0]
        logging.info(
            f"{self.dataset_name}: Starting with {initial_count} rows.")

        duplicate_count = self.dataset.duplicated(
            subset=[self.identifier_column], keep='first').sum()

        if duplicate_count > 0:
            logging.info(
                f"{self.dataset_name}: Found {duplicate_count} duplicate rows based on '{self.identifier_column}'.")

            self.dataset = self.dataset.drop_duplicates(
                subset=[self.identifier_column], keep='first')

            logging.info(
                f"{self.dataset_name}: Removed {duplicate_count} duplicates; {self.dataset.shape[0]} rows remain.")
        else:
            logging.info(
                f"{self.dataset_name}: No duplicates found based on '{self.identifier_column}'.")

        return self.dataset

    def save_processed_data(self):
        """Save the processed dataset to the specified output path.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.dataset.to_csv(self.output_path, index=False)
            logging.info(
                f"{self.dataset_name}: Data saved to {self.output_path}.")
            return True
        except Exception as e:
            logging.error(
                f"{self.dataset_name}: Error saving data to {self.output_path}: {e}")
            return False

    def process(self):
        """Process the dataset by applying all cleaning steps or load existing processed data.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        if not isinstance(self.dataset, pd.DataFrame):
            logging.error("DataPreprocessor expects a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame")

        # Check if processed file already exists
        if os.path.exists(self.output_path):
            logging.info(
                f"{self.dataset_name}: Processed file exists at {self.output_path}. Loading...")
            try:
                self.dataset = pd.read_csv(self.output_path)
                logging.info(
                    f"{self.dataset_name}: Loaded {self.dataset.shape[0]} rows from existing file.")
            except Exception as e:
                logging.error(
                    f"{self.dataset_name}: Error loading {self.output_path}: {e}")
                raise
        else:
            logging.info(
                f"{self.dataset_name}: Processing dataset with initial {self.dataset.shape[0]} rows")

            # Apply all preprocessing steps
            self.remove_unnamed_columns()
            self.remove_missing_values()
            self.remove_duplicate_records()

            logging.info(
                f"{self.dataset_name}: Processing complete. Final row count: {self.dataset.shape[0]}")

            # Save the processed dataset
            self.save_processed_data()

        return self.dataset
