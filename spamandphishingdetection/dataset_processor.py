import logging
import pandas as pd
import os


class DatasetProcessor:
    """
    A class to process datasets by removing unnamed columns, missing values, and duplicates, and saving the processed data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be processed.
    column_name : str
        The column name to check for duplicates.
    dataset_name : str
        The name of the dataset.
    save_path : str
        The path to save the processed data.
    """

    def __init__(self, df, column_name, dataset_name, save_path):
        self.df = df
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path

    def drop_unnamed_column(self):
        """
        Drop the 'Unnamed: 0' column if it exists in the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the 'Unnamed: 0' column removed if it existed.
        """
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
            logging.info(
                f"Dropped 'Unnamed: 0' column from {self.dataset_name}.")

        return self.df

    def check_and_remove_missing_values(self):
        """
        Check and remove missing values from the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values removed.
        """
        check_missing_values = self.df.isnull().sum()
        total_missing_values = check_missing_values.sum()
        logging.info(f"Total missing values: {total_missing_values}")
        logging.info(f"Removing missing values from {self.dataset_name}...")
        self.df = self.df.dropna()
        logging.info(
            f"Total number of rows after removing missing values from {self.dataset_name}: {self.df.shape[0]}")

        return self.df

    def remove_duplicates(self):
        """
        Remove duplicate rows based on the specified column.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with duplicates removed.
        """
        logging.info(f"Removing duplicate data....")

        # Log the initial number of rows
        initial_row_count = self.df.shape[0]
        logging.info(f"Initial number of rows: {initial_row_count}")

        # Identify duplicates
        num_duplicates_before = self.df.duplicated(
            subset=[self.column_name], keep=False).sum()
        logging.info(
            f"Total number of rows identified as duplicates based on '{self.column_name}': {num_duplicates_before}")

        # Remove duplicates
        self.df = self.df.drop_duplicates(
            subset=[self.column_name], keep='first')

        # Log the number of rows after removing duplicates
        final_row_count = self.df.shape[0]
        logging.info(
            f"Number of rows after removing duplicates: {final_row_count}")

        # Calculate the number of duplicates removed
        duplicates_removed = initial_row_count - final_row_count
        logging.info(
            f"Number of rows removed due to duplication: {duplicates_removed}")

        return self.df

    def save_processed_data(self):
        """
        Save the processed DataFrame to a CSV file.

        Returns
        -------
        None
        """
        try:
            self.df.to_csv(self.save_path, index=False)
            logging.info(f"Processed data saved to {self.save_path}\n")
        except PermissionError as e:
            logging.error(f"Permission denied: {e}")
        except Exception as e:
            logging.error(f"An error occurred while saving the file: {e}")

    def process_dataset(self):
        """
        Process the dataset by dropping unnamed columns, removing missing values, and removing duplicates.

        Returns
        -------
        pandas.DataFrame
            The processed DataFrame.
        """
        if os.path.exists(self.save_path):
            logging.info(
                f"Processed file already exists at {self.save_path}. Loading the file...")
            self.df = pd.read_csv(self.save_path)
        else:
            logging.info(
                f"Total number of rows in {self.dataset_name} DataFrame: {self.df.shape[0]}")
            self.drop_unnamed_column()
            self.check_and_remove_missing_values()
            self.remove_duplicates()
            logging.info(
                f"Total number of rows remaining in the {self.dataset_name}: {self.df.shape[0]}")
            logging.debug(
                f"{self.dataset_name} after removing duplicates: \n{self.df.head()}\n")
            self.save_processed_data()

        return self.df
