import json
import os
import logging
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from a JSON file based on the ENV variable.

    If config_path is not provided, it first attempts to load 'config.<env>.json' and falls back to 'config.json'.
    Raises FileNotFoundError if the file does not exist, or json.JSONDecodeError on invalid JSON.
    """
    env = os.environ.get("ENV", "development")
    if config_path is None:
        env_config_path = Path(f"config.{env}.json")
        if env_config_path.exists():
            config_path = env_config_path
        else:
            config_path = Path("config.json")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logging.error("Config file does not exist: %s", config_path)
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as e:
        logging.error(
            "Error decoding JSON from config file %s: %s", config_path, e)
        raise

    logging.info("Loaded config file: %s", config_path)
    return config


def ensure_directory_exists(path):
    """Ensure that the directory for the given path exists. If not, create it."""
    os.makedirs(path, exist_ok=True)


def check_config(config, required_keys):
    """Validate that all required keys exist in the config dictionary.

    Raises ValueError if any key is missing.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        logging.error("Missing required config keys: %s", missing)
        raise ValueError("Missing required keys: " + ", ".join(missing))
    logging.info(
        "Configuration validated successfully. All required keys are present.")
    return config


def get_file_paths(config):
    """Generate and ensure directories for required file paths based on the base_dir in config."""
    base_dir = config['base_dir']
    paths = {
        'ceas_08_dataset': Path(base_dir) / 'data' / 'ceas_08.csv',
        'preprocessed_spam_assassin_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_preprocessing' / 'preprocessed_spam_assassin.csv',
        'preprocessed_ceas_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_preprocessing' / 'preprocessed_ceas_08.csv',
        'extracted_spam_assassin_email_header_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'feature_engineering' / 'spam_assassin_extracted_email_header.csv',
        'extracted_ceas_email_header_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'feature_engineering' / 'ceas_extracted_email_header.csv',
        'merged_spam_assassin_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_integration' / 'merged_spam_assassin.csv',
        'merged_ceas_file': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_integration' / 'merged_ceas_08.csv',
        'merged_data_frame': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_integration' / 'merged_data_frame.csv',
        'cleaned_data_frame': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_cleaning' / 'cleaned_data_frame.csv',
        'cleaned_ceas_headers': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_cleaning' / 'cleaned_ceas_headers.csv',
        'merged_cleaned_ceas_headers': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_cleaning' / 'merged_cleaned_ceas_headers.csv',
        'merged_cleaned_data_frame': Path(base_dir) / 'output' / 'main_model_evaluation' / 'data_cleaning' / 'merged_cleaned_data_frame.csv',
        'noisy_data_frame': Path(base_dir) / 'output' / 'main_model_evaluation' / 'noise_injection' / 'noisy_data_frame.csv',
        'pipeline_path': Path(base_dir) / 'output' / 'main_model_evaluation' / 'feature_extraction'
    }

    for path in paths.values():
        ensure_directory_exists(path.parent)

    return {key: str(path) for key, path in paths.items()}


def get_model_path(config, fold_idx):
    """Construct the file path for the ensemble model for a given fold index."""
    base_dir = config['base_dir']
    return str(Path(base_dir) / 'output' / 'main_model_evaluation' / 'models_and_parameters' / f'Ensemble_Model_Fold_{fold_idx}.pkl')


def get_params_path(config, fold_idx):
    """Construct the file path for the best parameters JSON for a given fold index."""
    base_dir = config['base_dir']
    return str(Path(base_dir) / 'output' / 'main_model_evaluation' / 'models_and_parameters' / f'Best_Parameter_Fold_{fold_idx}.json')
