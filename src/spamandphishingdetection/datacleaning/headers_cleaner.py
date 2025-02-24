import os
import pandas as pd
import logging
import re


def get_email_headers(dataframe, csv_path, dataset_label='CEAS_08'):
    """Retrieve email header data from CSV, or process and save if not available.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing 'sender' and 'receiver' columns.
        csv_path (str): Path to CSV file for header data.
        dataset_label (str): Label for logging context.

    Returns:
        pd.DataFrame: DataFrame with header email addresses.

    Raises:
        ValueError: If required columns are missing.
    """
    if os.path.exists(csv_path):
        logging.info(f"CSV file {csv_path} exists. Loading header data...")
        try:
            headers_df = pd.read_csv(csv_path)
        except Exception as err:
            logging.error(f"Error reading {csv_path}: {err}")
            raise
    else:
        logging.info(
            f"CSV file {csv_path} not found. Processing header data for {dataset_label}...")
        headers_df = save_email_headers(dataframe, csv_path)
        logging.info(f"Header data processed and saved to {csv_path}.")
    return headers_df


def save_email_headers(dataframe, csv_path):
    """Extract and save email header addresses from the dataframe.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with 'sender' and 'receiver' columns.
        csv_path (str): Destination CSV file path.

    Returns:
        pd.DataFrame: DataFrame containing extracted header addresses.

    Raises:
        ValueError: If 'sender' or 'receiver' columns are missing.
    """
    # Check for required columns
    for col in ['sender', 'receiver']:
        if col not in dataframe.columns:
            raise ValueError(f"Missing required column: {col}")

    dataframe['sender'] = dataframe['sender'].apply(extract_email_address)
    dataframe['receiver'] = dataframe['receiver'].apply(extract_email_address)

    headers_df = dataframe[['sender', 'receiver']].copy()

    try:
        headers_df.to_csv(csv_path, index=False)
    except Exception as err:
        logging.error(f"Error saving header data to {csv_path}: {err}")
        raise

    return headers_df


def extract_email_address(raw_text):
    """Extract an email address from a string using defined regex patterns.

    Parameters:
        raw_text (str): The text to parse for an email address.

    Returns:
        str or None: The extracted email address if valid, else None.
    """
    if isinstance(raw_text, str):
        match = re.search(r'<([^>]+)>', raw_text)
        if match:
            candidate = match.group(1).strip()
            if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', candidate):
                return candidate
        candidate = raw_text.strip()
        if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', candidate):
            return candidate
    return None
