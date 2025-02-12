import logging
import pandas as pd
from datasets import load_dataset

from src.spamandphishingdetection import (
    load_config,
    check_config,
    get_file_paths,
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

    logging.info("Application is ready to proceed.")

    # Load the ceas dataset
    df_ceas = pd.read_csv(
        file_path['ceas_08_dataset'], sep=',', encoding='utf-8')
    
    # Load the spamassassin dataset
    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()

if __name__ == "__main__":
    main()
