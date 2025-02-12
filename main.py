import logging
import pandas as pd
from datasets import load_dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.spamandphishingdetection import (
    load_config,
    check_config,
    get_file_paths,
    log_label_percentages,
    DatasetProcessor,
    LabelMapper,
    build_ceas_pipeline,
    build_spamassassin_pipeline,
    initialize_environment
)


def main():
    # Initialize environment and logging
    log_path = initialize_environment(__file__)
    logging.info(
        "Environment setup complete. Log file created at: %s", log_path)

    # Load and validate the configuration
    config = load_config("config/config.json")
    required_keys = ["base_dir"]
    config = check_config(config, required_keys)
    file_path = get_file_paths(config)

    logging.info("Model Training begining...\n")

    # Load the ceas dataset
    df_ceas = pd.read_csv(
        file_path['ceas_08_dataset'], sep=',', encoding='utf-8')

    # Load the spamassassin dataset
    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()

    # Build pipelines for each dataset using the custom transformers.
    spam_pipeline = build_spamassassin_pipeline(
        file_path['preprocessed_spam_assassin_file'])
    ceas_pipeline = build_ceas_pipeline(file_path['preprocessed_ceas_file'])

    # * Data Preprocessing *#

    # Process the datasets through their respective pipelines.
    df_processed_spamassassin = spam_pipeline.fit_transform(df_spamassassin)
    df_processed_ceas = ceas_pipeline.fit_transform(df_ceas)

    # Optionally, combine dataframes after consistent processing
    combined_df = pd.concat([df_processed_spamassassin, df_processed_ceas])

    # Here you can call additional functions (for missing-value checks, etc.)
    # For example: check_missing_values(combined_df, 'Combined CEAS_08 and SpamAssassin')

if __name__ == "__main__":
    main()
