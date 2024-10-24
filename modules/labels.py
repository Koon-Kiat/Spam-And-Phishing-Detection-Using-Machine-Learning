import logging

label_descriptions = {
    0: "Safe",
    1: "Not Safe"
}


def log_label_percentages(df, dataset_name):
    """
    Logs the percentage of each label in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    dataset_name : str
        The name of the dataset being processed.

    Returns
    -------
    None
    """
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
