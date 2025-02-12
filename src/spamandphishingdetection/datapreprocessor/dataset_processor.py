import logging
import pandas as pd
import os


class DatasetProcessor:
    def __init__(self, df, column_name, dataset_name, save_path):
        self.df = df
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path

    def drop_unnamed_column(self):
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
            logging.info(
                f"Dropped 'Unnamed: 0' column from {self.dataset_name}.")

        return self.df

    def check_and_remove_missing_values(self):
        check_missing_values = self.df.isnull().sum()
        total_missing_values = check_missing_values.sum()
        logging.info(f"Total missing values: {total_missing_values}")
        self.df = self.df.dropna()
        logging.info(
            f"Total number of rows after removing missing values from {self.dataset_name}: {self.df.shape[0]}")

        return self.df

    def remove_duplicates(self):
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

        # Calculate the number of duplicates removed
        duplicates_removed = initial_row_count - final_row_count
        logging.info(
            f"Number of rows removed due to duplication: {duplicates_removed}")

        return self.df

    def save_processed_data(self):
        try:
            self.df.to_csv(self.save_path, index=False)
            logging.info(f"Processed data saved to {self.save_path}\n")
        except PermissionError as e:
            logging.error(f"Permission denied: {e}")
        except Exception as e:
            logging.error(f"An error occurred while saving the file: {e}")

    def process_dataset(self):

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
