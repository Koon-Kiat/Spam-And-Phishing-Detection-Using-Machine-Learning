import os
import pandas as pd
import logging
import re


def load_or_save_emails(df, output_file, df_name='CEAS_08'):
    """
    Load the cleaned email data from a CSV file or process and save the data if the file does not exist.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    output_file : str
        The file path to save or load the cleaned email data.
    df_name : str, optional
        The name of the DataFrame source. Default is 'CEAS_08'.

    Returns
    -------
    pandas.DataFrame
        The cleaned email data.
    """
    if os.path.exists(output_file):
        logging.info(
            f"Output file {output_file} already exists. Loading data from {output_file}...\n")
        df_cleaned = pd.read_csv(output_file)
    else:
        logging.info(
            f"Output file {output_file} does not exist. Loading data from {df_name}...")
        logging.info(f"Data loaded from {df_name}. Beginning processing...")

        df_cleaned = process_and_save_emails(df, output_file)

        logging.info(
            f"Data processing completed. Cleaned data saved to {output_file}.")

    return df_cleaned


def process_and_save_emails(df, output_file):
    """
    Process the DataFrame to extract sender and receiver emails and save to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    output_file : str
        The file path to save the processed email data.

    Returns
    -------
    pandas.DataFrame
        The DataFrame containing the extracted emails.
    """
    # Extract sender and receiver emails
    df['sender'] = df['sender'].apply(extract_email)
    df['receiver'] = df['receiver'].apply(extract_email)

    # Create a new DataFrame with the extracted emails
    email_df = df[['sender', 'receiver']]

    # Save the new DataFrame to a CSV file
    email_df.to_csv(output_file, index=False)
    return email_df


def extract_email(text):
    """
    Extract the email address from a given text.

    Parameters
    ----------
    text : str
        The text containing the email address.

    Returns
    -------
    str or None
        The extracted email address or None if no email address is found.
    """
    if isinstance(text, str):
        match = re.search(r'<([^>]+)>', text)
        if match:
            return match.group(1)
        elif re.match(r'^[^@]+@[^@]+\.[^@]+$', text):
            return text
    return None
