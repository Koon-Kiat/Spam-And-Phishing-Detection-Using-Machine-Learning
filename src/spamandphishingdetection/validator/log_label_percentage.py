import logging
import pandas as pd

label_descriptions = {
    0: "Safe",
    1: "Not Safe"
}


def log_label_percentages(df, dataset_name):
    """
    Logs label percentages for the given dataset.

    Parameters:
    df (pandas.DataFrame): DataFrame containing a 'label' column.
    dataset_name (str): Name of the dataset for logging context.
    """
    if not isinstance(df, pd.DataFrame):
        logging.error("log_label_percentages: Input is not a pandas DataFrame")
        raise ValueError("Input must be a pandas DataFrame")

    if 'label' not in df.columns:
        logging.warning(
            "log_label_percentages: 'label' column not found in DataFrame")
        return

    total_count = len(df)
    total_rows, total_columns = df.shape
    label_counts = df['label'].value_counts(normalize=True) * 100
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Total count: {total_count}")
    logging.info(f"Total rows: {total_rows}")
    logging.info(f"Total columns: {total_columns}")
    sorted_label_counts = label_counts.sort_index()
    num_labels = len(sorted_label_counts)
    for i, (label, percentage) in enumerate(sorted_label_counts.items()):
        description = label_descriptions.get(label, "Unknown")
        if i == num_labels - 1:
            logging.info(f"{description} percentage: {percentage:.2f}%\n")
        else:
            logging.info(f"{description} percentage: {percentage:.2f}%")
