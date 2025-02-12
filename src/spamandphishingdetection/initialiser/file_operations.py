import json
import os
import logging


def load_config(config_path=None):
    env = os.environ.get("ENV", "development")
    if config_path is None:
        env_config_path = f"config.{env}.json"
        if os.path.exists(env_config_path):
            config_path = env_config_path
        else:
            config_path = "config.json"

    if not os.path.exists(config_path):
        logging.error("Config file does not exist: %s", config_path)
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    logging.info("Loaded config file: %s", config_path)
    return config


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_config(config, required_keys):
    missing = [key for key in required_keys if key not in config]
    if missing:
        logging.error("Missing required config keys: %s", missing)
        raise ValueError("Missing required keys: " + ", ".join(missing))
    logging.info(
        "Configuration validated successfully. All required keys are present.")
    return config


def get_file_paths(config):
    base_dir = config['base_dir']
    file_paths = {
        'ceas_08_dataset': os.path.join(base_dir, 'data', 'ceas_08.csv'),
        'preprocessed_spam_assassin_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_preprocessing', 'preprocessed_spam_assassin.csv'),
        'preprocessed_ceas_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_preprocessing', 'preprocessed_ceas_08.csv'),
        'extracted_spam_assassin_email_header_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'feature_engineering', 'spam_assassin_extracted_email_header.csv'),
        'extracted_ceas_email_header_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'feature_engineering', 'ceas_extracted_email_header.csv'),
        'merged_spam_assassin_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_integration', 'merged_spam_assassin.csv'),
        'merged_ceas_file': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_integration', 'merged_ceas_08.csv'),
        'merged_data_frame': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_integration', 'merged_data_frame.csv'),
        'cleaned_data_frame': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_cleaning', 'cleaned_data_frame.csv'),
        'cleaned_ceas_headers': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_cleaning', 'cleaned_ceas_headers.csv'),
        'merged_cleaned_ceas_headers': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_cleaning', 'merged_cleaned_ceas_headers.csv'),
        'merged_cleaned_data_frame': os.path.join(base_dir, 'output', 'main_model_evaluation', 'data_cleaning', 'merged_cleaned_data_frame.csv'),
        'noisy_data_frame': os.path.join(base_dir, 'output', 'main_model_evaluation', 'noise_injection', 'noisy_data_frame.csv'),
        'pipeline_path': os.path.join(base_dir, 'output', 'main_model_evaluation', 'feature_extraction')
    }

    # Ensure directories exist
    for path in file_paths.values():
        ensure_directory_exists(os.path.dirname(path))

    return file_paths


def get_model_path(config, fold_idx):
    base_dir = config['base_dir']
    return os.path.join(base_dir, 'output', 'main_model_evaluation', 'models_and_parameters', f'Ensemble_Model_Fold_{fold_idx}.pkl')


def get_params_path(config, fold_idx):
    base_dir = config['base_dir']
    return os.path.join(base_dir, 'output', 'main_model_evaluation', 'models_and_parameters', f'Best_Parameter_Fold_{fold_idx}.json')
