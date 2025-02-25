"""
Data cleaning pipeline for spam and phishing email detection.

This module provides a pipeline class for cleaning and processing email text data,
coordinating the various steps of the cleaning process.
"""

import os
import pandas as pd
import logging


class DataCleaningPipeline:
    """
    A pipeline class to encapsulate the data cleaning process in modular steps.

    Steps:
    1. Load or clean data using the cleaning function.
    2. Combine the cleaned text with the original data.
    3. Verify the integrity of the combined DataFrame.
    4. Save the final combined DataFrame to CSV.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being processed.
    input_dataframe : pandas.DataFrame
        The input DataFrame containing the data to be cleaned.
    text_column_name : str
        The name of the column containing text data to be cleaned.
    cleaned_text_path : str
        File path to save/load the cleaned text data.
    final_output_path : str
        File path to save the final cleaned and combined DataFrame.
    cleaning_function : function
        The function to use for cleaning the data.
    """

    def __init__(self, dataset_name, input_dataframe, text_column_name,
                 cleaned_text_path, final_output_path, cleaning_function):
        self.dataset_name = dataset_name
        self.input_dataframe = input_dataframe
        self.text_column_name = text_column_name
        self.cleaned_text_path = cleaned_text_path
        self.final_output_path = final_output_path
        self.cleaning_function = cleaning_function

    def run(self):
        """
        Execute the data cleaning pipeline.

        Returns
        -------
        pandas.DataFrame
            The cleaned and combined DataFrame.
        """
        logging.info(
            f"Starting data cleaning pipeline for {self.dataset_name}")

        # Step 1: Load or clean text data
        cleaned_text_df = self._load_or_clean_data()

        # Step 2: Combine cleaned text with original DataFrame
        combined_df = self._combine_data(cleaned_text_df)

        # Step 3: Verify combined DataFrame
        self._verify_data_integrity(combined_df)

        # Step 4: Save cleaned DataFrame
        self._save_output(combined_df)

        logging.info(
            f"Data cleaning pipeline completed for {self.dataset_name}")
        return combined_df

    def _load_or_clean_data(self):
        """
        Load cleaned data from file if it exists, otherwise clean the data.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the cleaned text data.
        """
        if os.path.exists(self.cleaned_text_path):
            logging.info(
                f"Loading cleaned text data from {self.cleaned_text_path}")
            cleaned_df = pd.read_csv(self.cleaned_text_path)

            if 'cleaned_text' in cleaned_df.columns:
                cleaned_df['cleaned_text'] = cleaned_df['cleaned_text'].astype(
                    str).fillna('')
            else:
                logging.error(
                    f"Expected column 'cleaned_text' not found in {self.cleaned_text_path}")
                raise KeyError(
                    f"Column 'cleaned_text' missing in {self.cleaned_text_path}")

            texts = cleaned_df['cleaned_text'].tolist()
            if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
                raise ValueError("Loaded data is not a valid list of strings")

            return cleaned_df
        else:
            logging.info(f"Cleaning text data for {self.dataset_name}")
            return self.cleaning_function(
                self.dataset_name,
                self.input_dataframe,
                self.text_column_name,
                self.cleaned_text_path
            )

    def _combine_data(self, cleaned_text_df):
        """
        Combine the cleaned text data with the original DataFrame.

        Parameters
        ----------
        cleaned_text_df : pandas.DataFrame
            DataFrame containing the cleaned text data.

        Returns
        -------
        pandas.DataFrame
            Combined DataFrame with cleaned text and original features.
        """
        logging.info("Combining cleaned text with original data features")

        # Define required columns that should be in the input DataFrame
        required_cols = ['sender', 'receiver', 'date', 'urls', 'https_count', 'http_count', 'blacklisted_keywords_count',
                         'short_urls', 'has_ip_address', 'exclamation_count', 'uppercase_count', 'body_length',
                         'special_chars_count', 'label']

        # Check which required columns are actually present in the dataframe
        available_cols = [
            col for col in required_cols if col in self.input_dataframe.columns]
        missing_cols = [
            col for col in required_cols if col not in self.input_dataframe.columns]

        if missing_cols:
            logging.warning(
                f"Some columns were not found in the input DataFrame and will be skipped: {missing_cols}")

        if not available_cols:
            logging.error("No required columns found in input DataFrame")
            raise KeyError("No required columns found in input DataFrame")

        # Check if cleaned_text column exists
        if 'cleaned_text' not in cleaned_text_df.columns:
            logging.error(
                "Column 'cleaned_text' missing in cleaned text DataFrame")
            raise KeyError(
                "Column 'cleaned_text' missing in cleaned text DataFrame")

        # Reset indices to ensure proper alignment
        input_reset = self.input_dataframe.reset_index(drop=True)
        cleaned_reset = cleaned_text_df.reset_index(drop=True)

        # Concatenate the DataFrames
        combined_df = pd.concat([
            input_reset[available_cols],
            cleaned_reset[['cleaned_text']]
        ], axis=1)

        logging.info(
            f"Successfully combined DataFrames with columns: {list(combined_df.columns)}")
        return combined_df

    def _verify_data_integrity(self, combined_df):
        """
        Verify the integrity of the combined DataFrame.

        Parameters
        ----------
        combined_df : pandas.DataFrame
            The combined DataFrame to verify.

        Raises
        ------
        KeyError
            If required columns are missing.
        ValueError
            If label distributions don't match.
        """
        logging.info("Verifying data integrity of the combined DataFrame")

        if 'label' not in self.input_dataframe.columns or 'label' not in combined_df.columns:
            logging.error("Missing 'label' column in one of the DataFrames")
            raise KeyError("'label' column missing in DataFrames")

        # Check if the labels match
        orig_labels = set(self.input_dataframe['label'].unique())
        combined_labels = set(combined_df['label'].unique())
        if orig_labels != combined_labels:
            logging.error(
                f"Mismatch in labels. Original: {orig_labels}, Combined: {combined_labels}")
            raise ValueError(
                "Label sets do not match between original and combined DataFrames")

        # Check if the label distributions match
        orig_counts = self.input_dataframe['label'].value_counts().sort_index()
        combined_counts = combined_df['label'].value_counts().sort_index()
        if not orig_counts.equals(combined_counts):
            logging.error("Mismatch in label distributions")
            logging.error(f"Original distribution: \n{orig_counts}")
            logging.error(f"Combined distribution: \n{combined_counts}")
            raise ValueError(
                "Label distributions differ between original and combined DataFrames")

        logging.info("Data integrity verification passed")

    def _save_output(self, combined_df):
        """
        Save the combined DataFrame to CSV.

        Parameters
        ----------
        combined_df : pandas.DataFrame
            The DataFrame to save.
        """
        try:
            # Define the column order for the output file, including all required columns if present
            required_cols = ['sender', 'receiver', 'date', 'urls', 'https_count', 'http_count',
                             'blacklisted_keywords_count', 'short_urls', 'has_ip_address',
                             'exclamation_count', 'uppercase_count', 'body_length',
                             'special_chars_count', 'cleaned_text', 'label']

            # Get all available columns in the combined DataFrame
            available_cols = [
                col for col in required_cols if col in combined_df.columns]

            if 'cleaned_text' not in available_cols:
                logging.error(
                    "Missing critical column 'cleaned_text' in combined DataFrame")
                raise KeyError(
                    "Missing critical column 'cleaned_text' in combined DataFrame")

            if 'label' not in available_cols:
                logging.error(
                    "Missing critical column 'label' in combined DataFrame")
                raise KeyError(
                    "Missing critical column 'label' in combined DataFrame")

            logging.info(
                f"Saving DataFrame with these columns: {available_cols}")

            # Select columns in the specified order if they exist
            df_to_save = combined_df[available_cols]

            # Save to CSV
            df_to_save.to_csv(self.final_output_path, index=False)
            logging.info(
                f"Final cleaned data saved to {self.final_output_path}")
        except Exception as e:
            logging.error(
                f"Error saving final DataFrame to {self.final_output_path}: {e}")
            raise
