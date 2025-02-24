import logging
import pandas as pd
from datasets import load_dataset

from src.spamandphishingdetection import (
    load_config,
    check_config,
    get_file_paths,
    build_ceas_pipeline,
    build_spamassassin_pipeline,
    initialize_environment,
    feature_engineering,
    get_email_headers,
    integrate_datasets,
    verify_dataframe,
    get_merged_email_headers
)


def main():
    # * Environment Setup *#
    # Initialize environment and logging
    log_path = initialize_environment(__file__)
    logging.info(
        "Environment setup complete. Log file created at: %s", log_path)

    # Load and validate the configuration
    config = load_config("config/config.json")
    required_keys = ["base_dir"]
    config = check_config(config, required_keys)
    config_paths = get_file_paths(config)

    logging.info("Model Training begining...\n")

    # Load the ceas dataset
    df_ceas = pd.read_csv(
        config_paths['ceas_08_dataset'], sep=',', encoding='utf-8')

    # Load the spamassassin dataset
    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()

    # Build pipelines for each dataset using the custom transformers.
    spam_pipeline = build_spamassassin_pipeline(
        config_paths['preprocessed_spam_assassin_file'])
    ceas_pipeline = build_ceas_pipeline(config_paths['preprocessed_ceas_file'])

    # * Data Preprocessing *#
    processed_spamassassin_df = spam_pipeline.fit_transform(df_spamassassin)
    processed_ceas_df = ceas_pipeline.fit_transform(df_ceas)
    combined_df = pd.concat([processed_spamassassin_df, processed_ceas_df])

    # * Feature Engineering *#
    spam_assassin_features, ceas_features = feature_engineering(
        processed_spamassassin_df, processed_ceas_df, config_paths)

    # * Processing CEAS Dataset *#
    logging.info(
        f"Beginning Data Cleaning of CEAS_08 ['sender', 'receiver']...")
    # Use the merged header function which now handles file existence check and saving
    merged_ceas_df = get_merged_email_headers(
        processed_ceas_df,
        ceas_features,
        config_paths['cleaned_ceas_headers'],
        config_paths['merged_cleaned_ceas_headers']
    )

    logging.info(f"Merged CEAS dataset row count: {len(merged_ceas_df)}")

    # Verify CEAS Integration
    verify_dataframe(merged_ceas_df, len(processed_ceas_df), [
                     'sender', 'receiver'], "CEAS Integration")

    logging.info(
        f"Data cleaning of CEAS_08 ['sender', 'receiver'] completed.\n")

    # * Data Integration *#
    logging.info(
        "Beginning data integration of CEAS and SpamAssassin datasets...")
    integrated_df = integrate_datasets(
        processed_ceas_df,
        processed_spamassassin_df,
        ceas_features,
        spam_assassin_features,
        config_paths
    )
    logging.info(
        f"Data integration completed. Final integrated dataset row count: {len(integrated_df)}")


if __name__ == "__main__":
    main()
