import os
import pandas as pd
import logging
import re


def get_email_headers(df, headers_csv_path, dataset_label='CEAS_08'):
    """Retrieve email header data from CSV if available; otherwise, process and save header data.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'sender' and 'receiver' columns.
        headers_csv_path (str): Path to CSV file for header data.
        dataset_label (str): Label for logging context.

    Returns:
        pd.DataFrame: DataFrame with extracted header email addresses.

    Raises:
        ValueError: If required columns are missing.
    """
    if os.path.exists(headers_csv_path):
        logging.info(f"CSV file {headers_csv_path} exists. Attempting to load header data...")
        try:
            headers_data = pd.read_csv(headers_csv_path)
            # Check if the required columns are present
            if not all(col in headers_data.columns for col in ['sender', 'receiver']):
                logging.warning(f"CSV file {headers_csv_path} is missing required columns. Reprocessing header data for {dataset_label}...")
                headers_data = save_email_headers(df, headers_csv_path)
            else:
                logging.info(f"Header data loaded successfully from {headers_csv_path}.")
        except Exception as err:
            logging.error(f"Error reading {headers_csv_path}: {err}. Reprocessing header data for {dataset_label}...")
            headers_data = save_email_headers(df, headers_csv_path)
    else:
        logging.info(f"CSV file {headers_csv_path} not found. Processing header data for {dataset_label}...")
        headers_data = save_email_headers(df, headers_csv_path)
        logging.info(f"Header data processed and saved to {headers_csv_path}.")
    return headers_data


def save_email_headers(df, headers_csv_path):
    """Extract and save email header addresses from the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'sender' and 'receiver' columns.
        headers_csv_path (str): Destination CSV file path.

    Returns:
        pd.DataFrame: DataFrame containing extracted header addresses.

    Raises:
        ValueError: If 'sender' or 'receiver' columns are missing.
    """
    # Verify required columns exist
    for col in ['sender', 'receiver']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract email addresses using the helper function
    df['sender'] = df['sender'].apply(extract_email_address)
    df['receiver'] = df['receiver'].apply(extract_email_address)

    headers_data = df[['sender', 'receiver']].copy()

    try:
        headers_data.to_csv(headers_csv_path, index=False)
        logging.info(f"Header data successfully saved to {headers_csv_path}.")
    except Exception as err:
        logging.error(f"Error saving header data to {headers_csv_path}: {err}")
        raise

    return headers_data


def extract_email_address(raw_text):
    """Extract an email address from a string using regex patterns.

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


def get_merged_email_headers(processed_df, ceas_features, headers_csv_path, merged_csv_path, dataset_label='CEAS_08'):
    """Retrieve merged email header data if available; otherwise, process headers, merge with processed data and features, and save.
    
    Parameters:
        processed_df (pd.DataFrame): Processed CEAS DataFrame which originally contains header columns.
        ceas_features (pd.DataFrame): DataFrame of additional CEAS features to be merged.
        headers_csv_path (str): Path to CSV file for cleaned header data.
        merged_csv_path (str): Path to CSV file for merged header and processed data.
        dataset_label (str): Label for logging context.
        
    Returns:
        pd.DataFrame: Merged DataFrame containing headers, processed data, and features.
    """
    if os.path.exists(merged_csv_path):
        logging.info(f"Merged file {merged_csv_path} exists. Loading merged header data for {dataset_label}...")
        try:
            merged_data = pd.read_csv(merged_csv_path)
            logging.info(f"Merged header data loaded successfully from {merged_csv_path}.")
            return merged_data
        except Exception as err:
            logging.error(f"Error reading merged file {merged_csv_path}: {err}. Recreating merged data...")
    else:
        logging.info(f"Merged file {merged_csv_path} not found. Creating merged header data for {dataset_label}...")

    # Load or process header data
    header_data = get_email_headers(processed_df, headers_csv_path, dataset_label)

    if len(header_data) != len(processed_df):
        logging.error(f"Row count mismatch between header data and processed data for {dataset_label}.")
        raise ValueError(f"Row count mismatch between header data and processed data for {dataset_label}.")

    # Drop existing header columns from processed_df if present
    processed_df = processed_df.drop(columns=['sender', 'receiver'], errors='ignore')

    logging.info(f"Cleaned header columns: {header_data.columns.tolist()}")
    logging.info(f"Processed data columns (after dropping headers): {processed_df.columns.tolist()}")

    # Merge header data with processed_df and ceas_features
    merged_df = pd.concat([
        header_data.reset_index(drop=True),
        processed_df.reset_index(drop=True),
        ceas_features.reset_index(drop=True)
    ], axis=1)

    missing_records = merged_df[(merged_df['sender'].isnull()) | (merged_df['receiver'].isnull())]
    logging.info(f"Missing records in merged data: {len(missing_records)}")
    logging.info(f"Total rows in processed data: {len(processed_df)}")
    logging.info(f"Total rows in merged data: {len(merged_df)}")

    try:
        merged_df.to_csv(merged_csv_path, index=False)
        logging.info(f"Merged header data successfully saved to {merged_csv_path}")
    except Exception as err:
        logging.error(f"Error saving merged header data to {merged_csv_path}: {err}")
        raise

    return merged_df
