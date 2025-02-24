import logging


def check_missing_values(df, df_name, num_rows=1):
    missing_values = df.isnull().sum()
    total_missing_values = missing_values.sum()
    if total_missing_values == 0:
        logging.info(f"No missing values in {df_name}.")
    else:
        logging.info(
            f"Total missing values in {df_name}: {total_missing_values}")
        columns_with_missing = missing_values[missing_values > 0]
        logging.info(f"Columns with missing values in {df_name}:")
        for column, count in columns_with_missing.items():
            logging.info(f"Column '{column}': {count} missing values")
        rows_with_missing = df[df.isnull().any(axis=1)]
        if rows_with_missing.empty:
            logging.info(
                f"No rows with missing values found in {df_name} after initial check.")
