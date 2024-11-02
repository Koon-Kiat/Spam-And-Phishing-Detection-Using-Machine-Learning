# Standard Libraries
import os  # Interact with the operating system
import logging  # Logging library
from transformers.utils import logging as transformers_logging
import warnings  # Warning control
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import nltk  # Natural Language Toolkit
import spacy  # NLP library
import re  # Regular expressions
from tqdm import tqdm  # Progress bar
import joblib  # Joblib library
# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tabulate import tabulate  # Pretty-print tabular data
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import email  # Email handling
import email  # Email handling
import email.policy  # Email policies
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from imblearn.over_sampling import SMOTE  # Handling imbalanced data
import tensorflow as tf  # TensorFlow library
from bs4 import MarkupResemblesLocatorWarning  # HTML and XML parsing
from email.message import EmailMessage  # Email message
from email.parser import BytesParser  # Email parser
from typing import Dict, List, Union  # Type hints
import pickle  # Serialization library
from sklearn.decomposition import PCA  # Dimensionality reduction
from src.spamandphishingdetection import (
    initialize_environment,
    DatasetProcessor,
    count_urls,
    log_label_percentages,
    load_or_clean_data,
    data_cleaning,
    BERTFeatureExtractor,
    BERTFeatureTransformer,
)
from src.evaluationonthirddataset import (
    load_config,
    get_file_paths,
    load_or_extract_headers,
    EmailHeaderExtractor,
    run_pipeline_or_load
)


def main():
    nlp, loss_fn = initialize_environment(__file__)
    config = load_config()
    file_paths = get_file_paths(config)
    df_evaluation = pd.read_csv(file_paths['dataset'])

    # ****************************** #
    #       Data Preprocessing       #
    # ****************************** #
    logging.info("Beginning Data Preprocessing...")

    # Rename 'Email Type' column to 'Label' and map the values
    df_evaluation['label'] = df_evaluation['Email Type'].map(
        {'Safe Email': 0, 'Phishing Email': 1})
    df_evaluation = df_evaluation.rename(columns={'Email Text': 'text'})

    # Drop the original 'Email Type' column if no longer needed
    df_evaluation = df_evaluation.drop(columns=['Email Type'])

    processor_evaluation = DatasetProcessor(
        df_evaluation, "text", "Evaluation Dataset", file_paths['preprocessed_evaluation_dataset'])
    df_processed_evaluation = processor_evaluation.process_dataset()

    log_label_percentages(df_processed_evaluation, 'Evaluation Dataset')
    logging.info("Data Preprocessing completed.\n")

    # ****************************** #
    #       Feature Engineering      #
    # ****************************** #
    logging.info("Beginning Feature Engineering...")
    evaluation_email_headers = load_or_extract_headers(
        df_processed_evaluation, file_paths['extracted_evaluation_header_file'], EmailHeaderExtractor, 'Evaluation')
    logging.info(
        "Email header extraction and saving from Evaluation completed.")
    evaluation_email_headers['urls'] = evaluation_email_headers['texturls'].apply(
        count_urls)
    evaluation_email_headers.drop(
        columns=['mailto'], inplace=True)  # Drop the 'mailto' column
    # Drop the 'texturls' column
    evaluation_email_headers.drop(columns=['texturls'], inplace=True)
    logging.info("Feature Engineering completed.\n")

    # ****************************** #
    #       Data Integration         #
    # ****************************** #
    logging.info("Beginning Data Integration...")
    df_processed_evaluation.reset_index(inplace=True)
    evaluation_email_headers.reset_index(inplace=True)
    # evaluation_email_headers.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
    if len(df_processed_evaluation) == len(evaluation_email_headers):
        # Merge dataframes
        merged_evaluation = pd.merge(
            df_processed_evaluation, evaluation_email_headers, on='index', how='left')
        # Rename and reorder columns
        merged_evaluation = merged_evaluation.rename(columns={'text': 'body'})
        merged_evaluation = merged_evaluation[['sender', 'receiver', 'https_count', 'http_count',
                                               'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']]

        # Log missing rows
        missing_in_merged = merged_evaluation[merged_evaluation['index'].isnull(
        )]
        logging.info(
            f"Number of missing rows in Evaluation Dataframe: {len(missing_in_merged)}")
        logging.info(
            f'Total rows in Processed Evaluation Dataframe: {len(df_processed_evaluation)}')
        logging.info(
            f"Total rows in Evaluation Dataframe: {len(merged_evaluation)}")

        # Drop index column
        merged_evaluation.drop(columns=['index'], inplace=True)
    else:
        logging.error(
            "Length of the two dataframes do not match. Please check the dataframes.")
        raise ValueError(
            "Length of the two dataframes do not match. Please check the dataframes.")

    if len(merged_evaluation) != len(df_processed_evaluation):
        logging.error(
            "The number of rows in the Merge Evaluation Dataframe does not match the Processed Evaluation Dataframe.")
        raise ValueError(
            "The number of rows in the Merge Evaluation Dataframe does not match the Processed Evaluation Dataframe.")
    else:
        logging.info(
            "The number of rows in the Merge Evaluation Dataframe matches the Processed Evaluation Dataframe.")
        merged_evaluation.to_csv(
            file_paths['merged_evaluation_file'], index=False)
        logging.info(
            f"Data successfully saved to: {file_paths['merged_evaluation_file']}")
    logging.info("Data Integration completed.\n")

    # ************************* #
    #       Data Cleaning       #
    # ************************* #
    logging.info("Beginning Data Cleaning...")
    df_evaluation_clean = load_or_clean_data(
        "Evaluation", merged_evaluation, 'body', file_paths['cleaned_evaluation_data_frame'], data_cleaning)
    logging.info(f"Data Cleaning completed.\n")

    merged_evaluation_reset = merged_evaluation.reset_index(drop=True)
    df_evaluation_clean_reset = df_evaluation_clean.reset_index(drop=True)
    df_evaluation_clean_combined = pd.concat([
        merged_evaluation_reset[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
                                 'short_urls', 'has_ip_address', 'urls', 'label']],  # Select necessary columns from merged
        # Select the cleaned_text and label from df_clean
        df_evaluation_clean_reset[['cleaned_text']]
    ], axis=1)

    merged_evaluation_labels = merged_evaluation['label'].unique()
    df_evaluation_clean_combined_labels = df_evaluation_clean_combined['label'].unique(
    )
    if set(merged_evaluation_labels) != set(df_evaluation_clean_combined_labels):
        logging.error(f"Labels in Combined DataFrame do not match those in Cleaned Combined DataFrame. "
                      f"Combined DataFrame labels: {merged_evaluation_labels}, "
                      f"Cleaned Combined DataFrame labels: {df_evaluation_clean_combined_labels}")
        raise ValueError(
            "Labels do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info(
            "Labels in Combined DataFrame match those in Cleaned Combined DataFrame.")
    merged_evaluation_label_counts = merged_evaluation['label'].value_counts(
    ).sort_index()
    df_evaluation_clean_combined_counts = df_evaluation_clean_combined['label'].value_counts(
    ).sort_index()
    if not merged_evaluation_label_counts.equals(df_evaluation_clean_combined_counts):
        logging.error(
            "Label distributions in Combined DataFrame do not match those in Cleaned Combined DataFrame.")
        logging.error(
            f"Combined DataFrame distributions:\n{merged_evaluation_label_counts}")
        logging.error(
            f"Cleaned Combined DataFrame distributions:\n{df_evaluation_clean_combined_counts}")
        raise ValueError(
            "Label distributions do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info(
            "Label distributions in Combined DataFrame match those in Cleaned Combined DataFrame.")

        # Final columns to keep
        df_evaluation_clean_combined = df_evaluation_clean_combined[[
            'sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']]
        logging.info(
            f"Final combined DataFrame has {len(df_evaluation_clean_combined)} rows and columns: {df_evaluation_clean_combined.columns.tolist()}")
        df_evaluation_clean_combined.to_csv(
            file_paths['merged_cleaned_data_frame'], index=False)
        logging.info(f"Data Cleaning completed.\n")
        numerical_columns = ['https_count', 'http_count',
                             'blacklisted_keywords_count', 'urls', 'short_urls']
        for col in numerical_columns:
            df_evaluation_clean_combined[col] = pd.to_numeric(
                df_evaluation_clean_combined[col], errors='coerce').fillna(0)

    # ***************************** #
    #       Model Prediction        #
    # ***************************** #
    categorical_columns = ['sender', 'receiver']
    numerical_columns = ['https_count', 'http_count',
                         'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
    text_column = 'cleaned_text'

    for col in numerical_columns:
        df_evaluation_clean_combined[col] = pd.to_numeric(
            df_evaluation_clean_combined[col], errors='coerce').fillna(0)

    # Initialize BERT feature extractor and transformer
    bert_extractor = BERTFeatureExtractor()
    bert_transformer = BERTFeatureTransformer(feature_extractor=bert_extractor)

    # Define preprocessor for categorical and numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), categorical_columns),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_columns)
        ],
        remainder='passthrough'  # Keep other columns unchanged, like 'cleaned_text' and 'label'
    )

    # Define pipeline with preprocessor, BERT, SMOTE, and PCA
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('bert_features', bert_transformer),  # Custom transformer for BERT
        ('pca', PCA(n_components=777))  # Keep 95% of variance
    ])

    # Directory where you want to save the processed data
    data_dir = file_paths['data_dir']

    # Assuming df_evaluation_clean_combined is the DataFrame you are working with
    X = df_evaluation_clean_combined.drop(columns=['label'])
    y = df_evaluation_clean_combined['label']

    # Run the pipeline or load processed data
    processed_data, processed_labels = run_pipeline_or_load(
        X, y, pipeline, data_dir)

    # Load the saved model
    Main_Model = file_paths['main_model']
    Base_Model = file_paths['base_model']
    Base_Model_Optuna = file_paths['base_model_optuna']
    Stacked_Model_Optuna = file_paths['stacked_model_optuna']

    # List of folders containing models
    model_folders = [Main_Model, Base_Model,
                     Base_Model_Optuna, Stacked_Model_Optuna]

    # Placeholder for results
    results = []

    for folder in model_folders:
        folder_name = os.path.basename(folder)
        for model_file in os.listdir(folder):
            model_path = os.path.join(folder, model_file)
            if os.path.isfile(model_path):
                try:
                    saved_model = joblib.load(model_path)
                    model_name = model_file.replace('.pkl', '')
                    print(f"Evaluating model: {model_name}")
                    y_pred = saved_model.predict(processed_data)
                    test_accuracy = accuracy_score(processed_labels, y_pred)
                    confusion = confusion_matrix(processed_labels, y_pred)
                    classification = classification_report(processed_labels, y_pred, target_names=[
                                                           'Safe', 'Not Safe'], zero_division=1)
                    results.append({
                        'Model': model_name,
                        'Folder': folder_name,
                        'Accuracy': f"{test_accuracy * 100:.2f}%",
                        'Confusion Matrix': confusion,
                        'Classification Report': classification
                    })
                except Exception as e:
                    logging.error(
                        f"Error loading or evaluating model {model_file}: {e}")
            else:
                print(f"Skipping invalid path: {model_path}")

    # Convert results to a DataFrame for better formatting
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    output_path = file_paths['output_path']
    results_df[['Model', 'Folder', 'Accuracy']].to_csv(
        output_path, index=False)

    # Print the results in a table format
    print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))


if __name__ == '__main__':
    main()
