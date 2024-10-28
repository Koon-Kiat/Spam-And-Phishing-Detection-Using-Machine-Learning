import os
import json


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
        'dataset': os.path.join(base_dir, 'datasets', 'phishing_email.csv'),
        'preprocessed_evaluation_dataset': os.path.join(
            base_dir, 'third_dataset_evaluation', 'data_preprocessing', 'preprocessed_evaluation_dataset.csv'),
        'extracted_evaluation_header_file': os.path.join(
            base_dir, 'third_dataset_evaluation', 'feature_engineering', 'extracted_evaluation_header_file.csv'),
        'cleaned_evaluation_data_frame': os.path.join(
            base_dir, 'third_dataset_evaluation', 'data_cleaning', 'cleaned_evaluation_data_frame.csv'),
        'merged_evaluation_file': os.path.join(
            base_dir, 'third_dataset_evaluation', 'data_integration', 'merged_evaluation.csv'),
        'merged_cleaned_data_frame': os.path.join(
            base_dir, 'third_dataset_evaluation', 'data_cleaning', 'merged_cleaned_data_frame.csv'),
        'main_model': os.path.join(base_dir, 'data_pipeline', 'models_and_parameters'),
        'base_model': os.path.join(
            base_dir, 'additional_model_training', 'base_models'),
        'base_model_optuna': os.path.join(
            base_dir, 'additional_model_training', 'base_models_optuna'),
        'stacked_model_optuna': os.path.join(
            base_dir, 'additional_model_training', 'stacked_models'),
        'data_dir': os.path.join(base_dir, 'third_dataset_evaluation', 'feature_extraction'),
        'output_path': os.path.join(
            base_dir, 'third_dataset_evaluation', 'model_evaluation_result.csv')
    }

    # Ensure directories exist
    for path in file_paths.values():
        ensure_directory_exists(os.path.dirname(path))

    return file_paths
