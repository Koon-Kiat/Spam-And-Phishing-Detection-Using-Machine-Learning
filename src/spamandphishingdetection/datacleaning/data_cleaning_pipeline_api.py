"""
High-level API functions for the data cleaning pipeline.

This module provides simplified functions to access the data cleaning functionality,
making it easier to use without directly interacting with the underlying classes.
"""

import logging
import os
import pandas as pd

from src.spamandphishingdetection.datacleaning.data_cleaning_pipeline import DataCleaningPipeline

# Import the class at function level to avoid circular imports
def _get_text_processor():
    from src.spamandphishingdetection.datacleaning.text_processor import TextProcessor
    return TextProcessor()


def process_text_cleaning(dataset_name, dataframe, text_column, output_file):
    """
    Process and clean the text data in the specified column of a DataFrame.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being processed.
    dataframe : pandas.DataFrame
        The DataFrame containing the data to be cleaned.
    text_column : str
        The name of the column containing text data to be cleaned.
    output_file : str
        The file path where the cleaned data will be saved.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the cleaned text.
    """
    logging.info(f"Starting text cleaning for {dataset_name}")

    # Validate input dataframe contains necessary columns
    if text_column not in dataframe.columns:
        logging.error(f"Column '{text_column}' not found in DataFrame")
        raise KeyError(f"Column '{text_column}' not present in DataFrame")

    # Check for label column
    label_series = dataframe['label'] if 'label' in dataframe.columns else None
    if label_series is None:
        logging.warning(
            "Column 'label' not found in DataFrame. Proceeding without label information")

    # Initialize text processor and clean the text
    processor = _get_text_processor()
    try:
        cleaned_df = processor.clean_text(dataframe[text_column], label_series)
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        raise

    # Save cleaned text to file
    try:
        processor.save_cleaned_text(cleaned_df, output_file)
    except Exception as e:
        logging.error(f"Error saving cleaned text to {output_file}: {e}")
        raise

    logging.info(f"Text cleaning completed for {dataset_name}")
    return cleaned_df


def run_data_cleaning_pipeline(dataset_name, input_dataframe, text_column,
                               cleaned_text_path, final_output_path):
    """
    Execute the complete data cleaning pipeline for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being processed.
    input_dataframe : pandas.DataFrame
        The input DataFrame containing the data to be cleaned.
    text_column : str
        The name of the column containing text data to be cleaned.
    cleaned_text_path : str
        File path to save/load the cleaned text data.
    final_output_path : str
        File path to save the final cleaned and combined DataFrame.

    Returns
    -------
    pandas.DataFrame
        The cleaned and combined DataFrame.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(cleaned_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    pipeline = DataCleaningPipeline(
        dataset_name=dataset_name,
        input_dataframe=input_dataframe,
        text_column_name=text_column,
        cleaned_text_path=cleaned_text_path,
        final_output_path=final_output_path,
        cleaning_function=process_text_cleaning
    )

    return pipeline.run()


def clean_email_text(email_text):
    """
    Clean a single email text string without saving to file.

    This is a convenience function for cleaning individual emails,
    useful for API or interactive applications.

    Parameters
    ----------
    email_text : str
        The email text to clean.

    Returns
    -------
    str
        The cleaned email text.
    """
    processor = _get_text_processor()
    result = processor.clean_text([email_text])
    return result['cleaned_text'].iloc[0]
