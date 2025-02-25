"""
Email headers cleaning pipeline for managing the process of cleaning and merging header data.

This module provides a pipeline class for handling the complete email headers
cleaning process, including loading, cleaning, and merging header data with
processed features.
"""

import os
import pandas as pd
import logging

from src.spamandphishingdetection.datacleaning.headers_processor import EmailHeadersProcessor


class HeadersCleaningPipeline:
    """
    A pipeline class for managing email headers cleaning and merging.

    This pipeline handles:
    1. Loading or cleaning email header data
    2. Merging cleaned headers with additional features
    3. Validating the merged result
    4. Saving the final output

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being processed.
    input_dataframe : pandas.DataFrame
        The input DataFrame containing raw header data to be cleaned.
    features_dataframe : pandas.DataFrame
        DataFrame containing additional features to merge with cleaned headers.
    headers_output_path : str
        Path to save/load the cleaned header data.
    merged_output_path : str
        Path to save the merged output data.
    """

    def __init__(self, dataset_name, input_dataframe, features_dataframe,
                 headers_output_path, merged_output_path):
        self.dataset_name = dataset_name
        self.input_dataframe = input_dataframe
        self.features_dataframe = features_dataframe
        self.headers_output_path = headers_output_path
        self.merged_output_path = merged_output_path
        self.headers_processor = EmailHeadersProcessor(dataset_name)

    def run(self):
        """
        Execute the complete headers cleaning pipeline.

        Returns
        -------
        pandas.DataFrame
            The merged DataFrame with cleaned headers and features.
        """
        # Check if merged data already exists
        if os.path.exists(self.merged_output_path):
            logging.info(
                f"Loading existing merged data from {self.merged_output_path}")
            try:
                merged_data = pd.read_csv(self.merged_output_path)
                if self._validate_merged_data(merged_data):
                    logging.info(f"Loaded merged data validated successfully")
                    return merged_data
                else:
                    logging.warning(
                        "Loaded merged data failed validation. Reprocessing...")
            except Exception as e:
                logging.error(
                    f"Error loading merged data: {e}. Reprocessing...")

        # Step 1: Process header data
        cleaned_headers = self._process_header_data()

        # Step 2: Merge with features
        merged_result = self._merge_with_features(cleaned_headers)

        # Step 3: Validate and save
        if self._validate_merged_data(merged_result):
            self._save_merged_data(merged_result)

        return merged_result

    def _process_header_data(self):
        """
        Load header data if it exists, otherwise process and save it.

        Returns
        -------
        pandas.DataFrame
            DataFrame with processed header data.
        """
        if os.path.exists(self.headers_output_path):
            logging.info(
                f"Loading existing header data from {self.headers_output_path}")
            try:
                headers_data = pd.read_csv(self.headers_output_path)
                # Verify the loaded data has the expected columns
                if not all(col in headers_data.columns for col in ['sender', 'receiver']):
                    logging.warning(
                        "Loaded header data missing required columns. Reprocessing...")
                    headers_data = self.headers_processor.process_headers(
                        self.input_dataframe, self.headers_output_path)
                return headers_data
            except Exception as e:
                logging.error(
                    f"Error loading header data: {e}. Reprocessing...")

        return self.headers_processor.process_headers(self.input_dataframe, self.headers_output_path)

    def _merge_with_features(self, headers_df):
        """
        Merge cleaned headers with features DataFrame.

        Parameters
        ----------
        headers_df : pandas.DataFrame
            DataFrame with cleaned header data.

        Returns
        -------
        pandas.DataFrame
            Merged DataFrame with headers and features.
        """
        logging.info(f"Merging headers with features for {self.dataset_name}")

        # Remove existing header columns from input_dataframe if present
        df_without_headers = self.input_dataframe.drop(
            columns=['sender', 'receiver'], errors='ignore')

        # Reset indices to ensure proper alignment
        headers_reset = headers_df.reset_index(drop=True)
        input_reset = df_without_headers.reset_index(drop=True)
        features_reset = self.features_dataframe.reset_index(drop=True)

        # Concatenate the DataFrames
        merged_df = pd.concat([
            headers_reset,
            input_reset,
            features_reset
        ], axis=1)

        logging.info(f"Merged headers with features: {merged_df.shape}")
        return merged_df

    def _validate_merged_data(self, merged_df):
        """
        Validate the merged DataFrame.

        Parameters
        ----------
        merged_df : pandas.DataFrame
            The merged DataFrame to validate.

        Returns
        -------
        bool
            True if validation passed, False otherwise.
        """
        logging.info("Validating merged DataFrame")

        # Check if the merged DataFrame has the expected number of rows
        if len(merged_df) != len(self.input_dataframe):
            logging.error(
                f"Row count mismatch: {len(merged_df)} vs {len(self.input_dataframe)}")
            return False

        # Check if required columns are present
        required_cols = ['sender', 'receiver']
        missing_cols = [
            col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            logging.error(
                f"Missing columns in merged DataFrame: {missing_cols}")
            return False

        return True

    def _save_merged_data(self, merged_df):
        """
        Save the merged DataFrame to CSV.

        Parameters
        ----------
        merged_df : pandas.DataFrame
            The DataFrame to save.
        """
        try:
            os.makedirs(os.path.dirname(
                self.merged_output_path), exist_ok=True)
            merged_df.to_csv(self.merged_output_path, index=False)
            logging.info(f"Merged data saved to {self.merged_output_path}")
        except Exception as e:
            logging.error(
                f"Error saving merged data to {self.merged_output_path}: {e}")
            raise
