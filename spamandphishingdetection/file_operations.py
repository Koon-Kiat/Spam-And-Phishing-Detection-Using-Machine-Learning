import json
import os

def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_paths(config):
    base_dir = config['base_dir']
    file_paths = {
        'ceas_08_dataset': os.path.join(base_dir, 'datasets', 'ceas_08.csv'),
        'preprocessed_spam_assassin_file': os.path.join(base_dir, 'data_preprocessing', 'preprocessed_spam_assassin.csv'),
        'preprocessed_ceas_file': os.path.join(base_dir, 'data_preprocessing', 'preprocessed_ceas_08.csv'),
        'extracted_spam_assassin_email_header_file': os.path.join(base_dir, 'feature_engineering', 'spam_assassin_extracted_email_header.csv'),
        'extracted_ceas_email_header_file': os.path.join(base_dir, 'feature_engineering', 'ceas_extracted_email_header.csv'),
        'merged_spam_assassin_file': os.path.join(base_dir, 'data_integration', 'merged_spam_assassin.csv'),
        'merged_ceas_file': os.path.join(base_dir, 'data_integration', 'merged_ceas_08.csv'),
        'merged_data_frame': os.path.join(base_dir, 'data_integration', 'merged_data_frame.csv'),
        'cleaned_data_frame': os.path.join(base_dir, 'data_cleaning', 'cleaned_data_frame.csv'),
        'cleaned_ceas_headers': os.path.join(base_dir, 'data_cleaning', 'cleaned_ceas_headers.csv'),
        'merged_cleaned_ceas_headers': os.path.join(base_dir, 'data_cleaning', 'merged_cleaned_ceas_headers.csv'),
        'merged_cleaned_data_frame': os.path.join(base_dir, 'data_cleaning', 'merged_cleaned_data_frame.csv'),
        'noisy_data_frame': os.path.join(base_dir, 'noise_injection', 'noisy_data_frame.csv'),
        'pipeline_path': os.path.join(base_dir, 'feature_extraction')
    }

    # Ensure directories exist
    for path in file_paths.values():
        ensure_directory_exists(os.path.dirname(path))

    return file_paths

def get_model_path(config, fold_idx):
    base_dir = config['base_dir']
    return os.path.join(base_dir, 'models_and_parameters', f'Ensemble_Model_Fold_{fold_idx}.pkl')

def get_params_path(config, fold_idx):
    base_dir = config['base_dir']
    return os.path.join(base_dir, 'models_and_parameters', f'Best_Parameter_Fold_{fold_idx}.json')

