"""
API functions for email headers cleaning pipeline.

This module provides high-level functions to access the email headers cleaning functionality,
making it easier to use without directly interacting with the underlying classes.
"""

import logging
import os

from src.spamandphishingdetection.datacleaning.headers_pipeline import HeadersCleaningPipeline


def run_headers_cleaning_pipeline(dataset_name, input_dataframe, features_dataframe,
                                  headers_output_path, merged_output_path):
    """
    Process email headers for cleaning and merge with features.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being processed.
    input_dataframe : pandas.DataFrame
        DataFrame containing raw email headers.
    features_dataframe : pandas.DataFrame
        DataFrame containing features to merge with cleaned headers.
    headers_output_path : str
        Path to save/load the cleaned headers.
    merged_output_path : str
        Path to save the merged output.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with cleaned headers and features.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(headers_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)

    # Initialize and run the pipeline
    pipeline = HeadersCleaningPipeline(
        dataset_name=dataset_name,
        input_dataframe=input_dataframe,
        features_dataframe=features_dataframe,
        headers_output_path=headers_output_path,
        merged_output_path=merged_output_path
    )

    return pipeline.run()


def clean_single_email_header(header_text):
    """
    Clean a single email header field.

    This is a convenience function for cleaning individual email headers,
    useful for API or interactive applications.

    Parameters
    ----------
    header_text : str
        The raw email header text.

    Returns
    -------
    str
        The extracted email address or None if invalid.
    """
    from src.spamandphishingdetection.datacleaning.headers_utils import extract_email_address
    return extract_email_address(header_text)
