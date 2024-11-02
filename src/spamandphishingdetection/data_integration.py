import pandas as pd
import logging


def merge_dataframes(df1, df2, on_column='index', how='left', rename_columns=None, select_columns=None):
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    merged_df = pd.merge(df1, df2, on=on_column, how=how)
    if rename_columns:
        merged_df.rename(columns=rename_columns, inplace=True)
    if select_columns:
        merged_df = merged_df[select_columns]
    merged_df.drop(columns=[on_column], inplace=True)
    return merged_df


def verify_merged_dataframe(merged_df, original_df, dataset_name, file_path):
    if len(merged_df) != len(original_df):
        logging.error(
            f"The number of rows in the Merged {dataset_name} DataFrame does not match the original.")
        raise ValueError(
            f"The number of rows in the Merged {dataset_name} DataFrame does not match the original.")
    else:
        logging.info(
            f"The number of rows in the Merged {dataset_name} DataFrame matches the original.")
        merged_df.to_csv(file_path, index=False)
        logging.info(
            f"Merged {dataset_name} DataFrame successfully saved to {file_path}\n")


def combine_dataframes(df1, df2, common_columns):
    return pd.concat([df1[common_columns], df2[common_columns]])


def verify_combined_dataframe(combined_df, combined_percentage_df):
    combined_labels = set(combined_df['label'].unique())
    percentage_labels = set(combined_percentage_df['label'].unique())
    if combined_labels != percentage_labels:
        logging.error(f"Labels in Merged DataFrame do not match those in Combined CEAS_08 and SpamAssassin (No Processing). "
                      f"Merged DataFrame labels: {combined_labels}, "
                      f"Combined Processed DataFrame labels: {percentage_labels}")
        raise ValueError(
            "Labels do not match between Merged DataFrame and Combined CEAS_08 and SpamAssassin (No Processing).")
    else:
        logging.info(
            "Labels in Merged DataFrame match those in Combined CEAS_08 and SpamAssassin (No Processing).")

    combined_label_counts = combined_df['label'].value_counts().sort_index()
    percentage_label_counts = combined_percentage_df['label'].value_counts(
    ).sort_index()
    if not combined_label_counts.equals(percentage_label_counts):
        logging.error(
            "Label distributions in Merged DataFrame do not match those in Combined CEAS_08 and SpamAssassin (No Processing).")
        logging.error(
            f"Merged DataFrame distributions:\n{combined_label_counts}")
        logging.error(
            f"Combined CEAS_08 and SpamAssassin (No Processing) distributions:\n{percentage_label_counts}")
        raise ValueError(
            "Label distributions do not match between Merged DataFrame and Combined CEAS_08 and SpamAssassin (No Processing).")
    else:
        logging.info(
            "Label distributions in Merged DataFrame match those in Combined CEAS_08 and SpamAssassin (No Processing).")

    if len(combined_df) != len(combined_percentage_df):
        logging.error(
            "The number of rows in the Merged DataFrame does not match the Combined CEAS_08 and SpamAssassin (No Processing).")
        raise ValueError(
            "The number of rows in the Merged DataFrame does not match the Combined CEAS_08 and SpamAssassin (No Processing).")
    else:
        logging.info(
            "The number of rows in the Merged DataFrame matches the Combined CEAS_08 and SpamAssassin (No Processing).")


def save_combined_dataframe(combined_df, file_path):
    combined_df.to_csv(file_path, index=False)
    logging.info(f"Merged DataFrame successfully saved to {file_path}")
