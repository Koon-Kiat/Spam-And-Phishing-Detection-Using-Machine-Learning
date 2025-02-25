"""
Email headers processor for cleaning and standardizing email header data.

This module provides a class for cleaning email header fields like sender and receiver,
which is separate from the body text cleaning process.
"""

import os
import pandas as pd
import logging

from src.spamandphishingdetection.datacleaning.headers_utils import extract_email_address


class EmailHeadersProcessor:
    """
    A class for processing and cleaning email headers data.

    This class specifically handles extracting and standardizing email addresses
    from sender and receiver fields in email datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being processed.
    """

    def __init__(self, dataset_name="Email Dataset"):
        self.dataset_name = dataset_name
        logging.info(
            f"Initializing EmailHeadersProcessor for {dataset_name}...")

    def process_headers(self, dataframe, output_path=None):
        """
        Process email header fields to extract clean email addresses.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing email header fields (sender, receiver).
        output_path : str, optional
            Path to save the processed headers.

        Returns
        -------
        pandas.DataFrame
            DataFrame with cleaned email header fields.
        """
        logging.info(f"Processing email headers for {self.dataset_name}...")

        # Verify required columns exist
        for col in ['sender', 'receiver']:
            if col not in dataframe.columns:
                logging.error(f"Missing required column: {col}")
                raise ValueError(f"Missing required column: {col}")

        # Create a copy to avoid modifying the original
        result_df = dataframe[['sender', 'receiver']].copy()

        # Extract email addresses
        result_df['sender'] = result_df['sender'].apply(extract_email_address)
        result_df['receiver'] = result_df['receiver'].apply(
            extract_email_address)

        # Count valid email addresses
        valid_senders = result_df['sender'].notna().sum()
        valid_receivers = result_df['receiver'].notna().sum()
        total_rows = len(result_df)

        logging.info(
            f"Extracted {valid_senders}/{total_rows} valid sender addresses ({valid_senders/total_rows:.2%})")
        logging.info(
            f"Extracted {valid_receivers}/{total_rows} valid receiver addresses ({valid_receivers/total_rows:.2%})")

        # Save to file if path is provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result_df.to_csv(output_path, index=False)
                logging.info(f"Saved cleaned headers to {output_path}")
            except Exception as e:
                logging.error(
                    f"Error saving header data to {output_path}: {e}")
                raise

        return result_df
