import logging
import pandas as pd
import os


class DatasetProcessor:
    """Preprocess a dataset dataframe by removing unnamed columns, missing values, and duplicates.

    Parameters:
        data (pd.DataFrame): Input DataFrame to process.
        id_column (str): Column used to identify duplicates.
        dataset_name (str): Identifier for logging.
        out_path (str): File path to save the processed DataFrame.
    """

    def __init__(self, data, id_column, dataset_name, out_path):
        self.data = data
        self.id_column = id_column
        self.dataset_name = dataset_name
        self.out_path = out_path

    def drop_unnamed(self):
        """Remove the 'Unnamed: 0' column if found in the data."""
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.drop(columns=['Unnamed: 0'])
            logging.info(
                f"Dropped 'Unnamed: 0' column from {self.dataset_name}.")
        return self.data

    def remove_missing(self):
        """Log and drop rows containing missing values from the data."""
        missing = self.data.isnull().sum()
        total_missing = missing.sum()
        logging.info(
            f"{self.dataset_name}: {total_missing} missing values total.")
        self.data = self.data.dropna()
        logging.info(
            f"{self.dataset_name}: {self.data.shape[0]} rows remain after dropping missing values.")
        return self.data

    def remove_duplicates(self):
        """Drop duplicate rows based on the id column and log the changes."""
        start_count = self.data.shape[0]
        logging.info(f"{self.dataset_name}: Starting with {start_count} rows.")
        dup_count = self.data.duplicated(
            subset=[self.id_column], keep='first').sum()
        logging.info(
            f"{self.dataset_name}: Found {dup_count} duplicate rows based on '{self.id_column}'.")
        self.data = self.data.drop_duplicates(
            subset=[self.id_column], keep='first')
        end_count = self.data.shape[0]
        logging.info(
            f"{self.dataset_name}: Removed {start_count - end_count} duplicates; now {end_count} rows remain.")
        return self.data

    def save_data(self):
        """Save the processed data to the specified output path."""
        try:
            self.data.to_csv(self.out_path, index=False)
            logging.info(
                f"{self.dataset_name}: Data saved to {self.out_path}.\n")
        except Exception as e:
            logging.error(
                f"{self.dataset_name}: Error saving data to {self.out_path}: {e}")
            raise

    def process(self):
        """Process the dataset by dropping unnamed columns, removing missing values and duplicates, then save the data.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        if not isinstance(self.data, pd.DataFrame):
            logging.error("DatasetProcessor expects a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame")

        if os.path.exists(self.out_path):
            logging.info(
                f"{self.dataset_name}: Processed file exists at {self.out_path}. Loading...")
            try:
                self.data = pd.read_csv(self.out_path)
            except Exception as e:
                logging.error(
                    f"{self.dataset_name}: Error loading {self.out_path}: {e}")
                raise
        else:
            logging.info(
                f"{self.dataset_name}: Starting processing; initial rows: {self.data.shape[0]}")
            self.drop_unnamed()
            self.remove_missing()
            self.remove_duplicates()
            logging.info(
                f"{self.dataset_name}: Processed rows: {self.data.shape[0]}")
            self.save_data()

        return self.data
