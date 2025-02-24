import logging
import pandas as pd


def verify_dataframe(df, expected_rows, required_columns, step_name):
    """Verify the integrity of a dataframe by checking row count and required columns.

    Parameters:
        df (pd.DataFrame): DataFrame to verify.
        expected_rows (int): Expected number of rows.
        required_columns (list): List of columns that must be present in the DataFrame.
        step_name (str): Description of the integration step for logging.

    Raises:
        ValueError: If any verification fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"{step_name}: Provided object is not a pandas DataFrame")
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"{step_name}: Expected {expected_rows} rows, but got {df.shape[0]}")
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"{step_name}: Required column '{col}' is missing")
    logging.info(f"{step_name}: Dataframe verification passed.")
    return True
