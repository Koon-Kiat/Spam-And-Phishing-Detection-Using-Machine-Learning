import os
import logging
import pandas as pd
 
from src.spamandphishingdetection import get_email_headers
from src.spamandphishingdetection.validator.verify_dataframe import verify_dataframe


def _integrate_ceas(processed_ceas_df, ceas_features, config_paths):
    """
    Integrate the CEAS dataset by merging cleaned headers, processed data, and extracted features.
    """
    logging.info("Integrating CEAS dataset...")
    cleaned_ceas_headers_df = get_email_headers(
        processed_ceas_df, config_paths['cleaned_ceas_headers'])

    if len(cleaned_ceas_headers_df) != len(processed_ceas_df):
        logging.error(
            "CEAS Integration: Row count mismatch between cleaned headers and processed CEAS dataset.")
        raise ValueError(
            "CEAS Integration: Row count mismatch between cleaned headers and processed CEAS dataset.")

    # Drop header columns if present
    processed_ceas_df = processed_ceas_df.drop(
        columns=['sender', 'receiver'], errors='ignore')

    merged_ceas_df = pd.concat([
        cleaned_ceas_headers_df.reset_index(drop=True),
        processed_ceas_df.reset_index(drop=True),
        ceas_features.reset_index(drop=True)
    ], axis=1)

    # Verify integration by checking required header columns
    verify_dataframe(merged_ceas_df, len(processed_ceas_df), [
                     'sender', 'receiver'], "CEAS Integration")

    ceas_out_path = config_paths.get('merged_cleaned_ceas_headers')
    if ceas_out_path:
        merged_ceas_df.to_csv(ceas_out_path, index=False)
        logging.info(f"Merged CEAS dataset saved to {ceas_out_path}")

    return merged_ceas_df


def _integrate_spamassassin(processed_spamassassin_df, spamassassin_features, config_paths):
    """
    Integrate the SpamAssassin dataset by concatenating processed data and extracted features.
    """
    logging.info("Integrating SpamAssassin dataset...")
    spamassassin_merged_df = pd.concat([
        processed_spamassassin_df.reset_index(drop=True),
        spamassassin_features.reset_index(drop=True)
    ], axis=1)

    if len(spamassassin_merged_df) != len(processed_spamassassin_df):
        logging.error(
            "SpamAssassin Integration: Row count mismatch after merging features.")
        raise ValueError(
            "SpamAssassin Integration: Row count mismatch after merging features.")

    spam_out_path = config_paths.get('merged_spamassassin')
    if spam_out_path:
        spamassassin_merged_df.to_csv(spam_out_path, index=False)
        logging.info(f"Merged SpamAssassin dataset saved to {spam_out_path}")

    return spamassassin_merged_df


def integrate_datasets(processed_ceas_df, processed_spamassassin_df, ceas_features, spamassassin_features, config_paths):
    """
    Integrate both CEAS and SpamAssassin datasets and combine them.

    Steps:
      1. Integrate CEAS dataset.
      2. Integrate SpamAssassin dataset.
      3. Concatenate both datasets and perform final verification.

    Returns:
        Combined integrated DataFrame.
    """
    final_path = config_paths.get('merged_data_frame')

    if final_path and os.path.exists(final_path):
        logging.info(
            f"Final integrated dataset found at {final_path}. Loading...")
        try:
            integrated_df = pd.read_csv(final_path)
            return integrated_df
        except Exception as e:
            logging.error(f"Error loading final integrated dataset: {e}")
            raise

    ceas_df = _integrate_ceas(processed_ceas_df, ceas_features, config_paths)
    spam_df = _integrate_spamassassin(
        processed_spamassassin_df, spamassassin_features, config_paths)

    logging.info("Combining CEAS and SpamAssassin datasets...")
    combined_integrated_df = pd.concat([
        ceas_df,
        spam_df
    ], axis=0, ignore_index=True)
    expected_rows = len(ceas_df) + len(spam_df)
    verify_dataframe(combined_integrated_df, expected_rows,
                     [], "Final Integration")
    logging.info(
        f"Final integrated dataset row count: {len(combined_integrated_df)}")

    if final_path:
        combined_integrated_df.to_csv(final_path, index=False)
        logging.info(f"Final integrated dataset saved to {final_path}")
    else:
        logging.warning(
            "Final integrated dataset output path not provided in configuration.")

    return combined_integrated_df
