import logging
import pandas as pd
from datasets import load_dataset

from src.spamandphishingdetection import (
    load_config,
    check_config,
    get_file_paths,
    initialize_environment,
    feature_engineering,
    integrate_datasets,
    verify_dataframe,
    run_data_cleaning_pipeline,
    run_headers_cleaning_pipeline,
    build_spamassassin_pipeline,
    build_ceas_pipeline
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

    logging.info("Model Training beginning...\n")

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
    logging.info("Beginning data preprocessing...")
    processed_spamassassin_df = spam_pipeline.fit_transform(df_spamassassin)
    processed_ceas_df = ceas_pipeline.fit_transform(df_ceas)
    combined_df = pd.concat([processed_spamassassin_df, processed_ceas_df])
    logging.info(
        f"Data preprocessing complete. Combined dataset row count: {len(combined_df)}")

    # * Feature Engineering *#
    spam_assassin_features, ceas_features = feature_engineering(
        processed_spamassassin_df, processed_ceas_df, config_paths)

    # * Processing CEAS Dataset Headers *#
    logging.info(
        f"Beginning Data Cleaning of CEAS_08 email headers ['sender', 'receiver']...")

    # Use the new modular headers cleaning pipeline
    merged_ceas_df = run_headers_cleaning_pipeline(
        dataset_name='CEAS_08',
        input_dataframe=processed_ceas_df,
        features_dataframe=ceas_features,
        headers_output_path=config_paths['cleaned_ceas_headers'],
        merged_output_path=config_paths['merged_cleaned_ceas_headers']
    )

    logging.info(f"Merged CEAS dataset row count: {len(merged_ceas_df)}")

    # Verify CEAS Integration
    verify_dataframe(merged_ceas_df, len(processed_ceas_df), [
                     'sender', 'receiver'], "CEAS Headers Cleaning")

    logging.info(f"Data cleaning of CEAS_08 email headers completed.\n")

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
        f"Data integration completed. Final integrated dataset row count: {len(integrated_df)}\n")

    # * Data Cleaning for 'body' Text *#
    logging.info("Beginning text cleaning for email body content...")

    # Use the run_data_cleaning_pipeline function to process the body column
    cleaned_combined_df = run_data_cleaning_pipeline(
        dataset_name='Merged DataFrame',
        input_dataframe=integrated_df,
        text_column='body',
        cleaned_text_path=config_paths['cleaned_data_frame'],
        final_output_path=config_paths['merged_cleaned_data_frame']
    )

    logging.info(
        f"Text cleaning for email body completed. Cleaned dataset row count: {len(cleaned_combined_df)}")

    # Verify data cleaning results
    verify_dataframe(cleaned_combined_df, len(integrated_df),
                     ['cleaned_text', 'label'], "Text Cleaning")

    # * Noise Injection *#

    # * Data Splitting *#

    # * Feature Extraction and Data Imbalance Handling *#

    # * Model Training *#


if __name__ == "__main__":
    main()
