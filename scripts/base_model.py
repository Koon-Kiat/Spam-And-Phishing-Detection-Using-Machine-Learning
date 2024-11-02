import os  # Interact with the operating system
import logging  # Logging library
import sys  # System-specific parameters and functions
from transformers.utils import logging as transformers_logging
import warnings  # Warning control
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import nltk  # Natural Language Toolkit
import spacy  # NLP library
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Preprocessing
import email  # Email handling
import email.policy  # Email policies
from imblearn.over_sampling import SMOTE  # Handling imbalanced data
import tensorflow as tf  # TensorFlow library
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from xgboost import XGBClassifier  # XGBoost Classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
# K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier  # LightGBM Classifier
from bs4 import MarkupResemblesLocatorWarning  # HTML and XML parsing
from datasets import load_dataset  # Load datasets
from spamandphishingdetection import (
    initialize_environment,
    load_config,
    get_file_paths,
    get_model_path,
    get_params_path,
    DatasetProcessor,
    log_label_percentages,
    check_missing_values,
    feature_engineering,
    load_or_save_emails,
    process_and_save_emails,
    merge_dataframes,
    verify_merged_dataframe,
    combine_dataframes,
    verify_combined_dataframe,
    save_combined_dataframe,
    load_or_clean_data,
    data_cleaning,
    save_dataframe_to_csv,
    combine_columns_for_cleaning,
    generate_noisy_dataframe,
    stratified_k_fold_split,
    BERTFeatureExtractor,
    BERTFeatureTransformer,
    RareCategoryRemover,
    run_pipeline_or_load,
    base_model_training,
    plot_learning_curve
)

# Main processing function


def main():
    nlp, loss_fn = initialize_environment(__file__)
    config_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'config', 'config.json'))
    config = load_config(config_path)
    file_paths = get_file_paths(config)

    # Load the datasets
    df_ceas = pd.read_csv(
        file_paths['ceas_08_dataset'], sep=',', encoding='utf-8')
    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()

    try:
        # ****************************** #
        #       Data Preprocessing       #
        # ****************************** #
        logging.info(f"Beginning Data Preprocessing...")

        # Change label values to match the labeling scheme
        df_spamassassin['label'] = df_spamassassin['label'].map({1: 0, 0: 1})
        processor_spamassassin = DatasetProcessor(
            df_spamassassin, 'text', 'spam_assassin', file_paths['preprocessed_spam_assassin_file'])
        df_processed_spamassassin = processor_spamassassin.process_dataset()
        processor_ceas = DatasetProcessor(
            df_ceas, 'body', 'ceas_08', file_paths['preprocessed_ceas_file'])
        df_processed_ceas = processor_ceas.process_dataset()

        # Combined DataFrame
        combined_percentage_df = pd.concat(
            [df_processed_spamassassin, df_processed_ceas])

        log_label_percentages(df_processed_ceas, 'CEAS_08')
        log_label_percentages(df_processed_spamassassin, 'SpamAssassin')
        log_label_percentages(
            combined_percentage_df, 'Combined CEAS_08 and SpamAssassin (No Processing)')
        check_missing_values(combined_percentage_df,
                             'Combined CEAS_08 and SpamAssassin (No Processing)')
        logging.info(f"Data Preprocessing completed.\n")
        # Columns in CEAS_08 dataset: ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
        # Columns in SpamAssassin dataset: ['label', 'group', 'text']

        # ****************************** #
        #       Feature Engineering      #
        # ****************************** #
        logging.info(f"Beginning Feature Engineering...")

        spamassassin_headers_df, ceas_headers_df = feature_engineering(
            df_processed_spamassassin, df_processed_ceas, file_paths)

        # ************************* #
        #       Data Cleaning       #
        # ************************* #
        logging.info(
            f"Beginning Data Cleaning of CEAS_08 ['sender', 'receiver']...")
        df_cleaned_ceas_headers = load_or_save_emails(
            df_processed_ceas, file_paths['cleaned_ceas_headers'])

        logging.info(
            f"Beginning merging of Cleaned Headers of CEAS_08 with Processed CEAS_08...")
        if len(df_cleaned_ceas_headers) != len(df_processed_ceas):
            logging.error(
                "The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
            raise ValueError(
                "The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
        else:
            df_processed_ceas.drop(
                columns=['sender', 'receiver'], inplace=True)

            # Corrected logging statements below
            logging.info(
                f"Columns in df_cleaned_ceas_headers: {df_cleaned_ceas_headers.columns.tolist()}")
            logging.info(
                f"Columns in df_processed_ceas: {df_processed_ceas.columns.tolist()}")

            df_cleaned_ceas_headers_merge = pd.concat(
                [df_cleaned_ceas_headers.reset_index(drop=True),
                 df_processed_ceas.reset_index(drop=True)], axis=1)

            missing_in_cleaned_ceas_header_merged = df_cleaned_ceas_headers_merge[
                (df_cleaned_ceas_headers_merge['sender'].isnull()) |
                (df_cleaned_ceas_headers_merge['receiver'].isnull())]

            logging.info(
                f"Number of missing rows in Merged Cleaned Headers of CEAS_08 DataFrame: {len(missing_in_cleaned_ceas_header_merged)}")
            logging.info(
                f'Total rows in Processed CEAS_08 Dataframe: {len(df_processed_ceas)}')
            logging.info(
                f"Total rows in Merged Cleaned Headers of CEAS_08 Dataframe: {len(df_cleaned_ceas_headers_merge)}")

        if len(df_cleaned_ceas_headers_merge) != len(df_processed_ceas):
            logging.error(
                "The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
            raise ValueError(
                "The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.\n")
        else:
            logging.info(
                "The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame matches Processed CEAS_08.")
            df_cleaned_ceas_headers_merge.to_csv(
                file_paths['merged_cleaned_ceas_headers'], index=False)
            logging.info(
                f"Merged Cleaned Headers of CEAS_08 DataFrame successfully saved to {file_paths['merged_cleaned_ceas_headers']}")

        logging.info(
            f"Data Cleaning of CEAS_08 ['sender', 'receiver'] completed.\n")

        # ****************************** #
        #       Data Integration         #
        # ****************************** #
        logging.info(f"Beginning Data Integration...")

        # Merging Processed SpamAssassin dataset with the extracted information
        logging.info(
            f"Merging Processed Spam Assassin and Spam Assassin Header Dataframes...")
        merged_spamassassin_df = merge_dataframes(
            df_processed_spamassassin, spamassassin_headers_df, on_column='index',
            rename_columns={'text': 'body'},
            select_columns=['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
                            'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']
        )
        verify_merged_dataframe(merged_spamassassin_df, df_processed_spamassassin,
                                'Spam Assassin', file_paths['merged_spam_assassin_file'])

        # Merge Processed CEAS_08 dataset with the extracted information
        logging.info(
            f"Merging Processed CEAS_08 and CEAS_08 Header Dataframes...")
        merged_ceas_df = merge_dataframes(
            df_cleaned_ceas_headers_merge, ceas_headers_df, on_column='index',
            select_columns=['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
                            'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']
        )
        verify_merged_dataframe(
            merged_ceas_df, df_cleaned_ceas_headers_merge, 'CEAS_08', file_paths['merged_ceas_file'])

        # Merge Spam Assassin and CEAS_08 datasets
        logging.info(f"Merging Spam Assassin and CEAS_08 Dataframes...")
        common_columns = ['sender', 'receiver', 'https_count', 'http_count',
                          'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label']
        combined_df = combine_dataframes(
            merged_spamassassin_df, merged_ceas_df, common_columns)
        verify_combined_dataframe(combined_df, combined_percentage_df)
        save_combined_dataframe(combined_df, file_paths['merged_data_frame'])

        logging.info(f"Data Integration completed.\n")

        # ************************* #
        #       Data Cleaning       #
        # ************************* #
        logging.info(f"Beginning Data Cleaning ['body']...")
        df_clean_body = load_or_clean_data(
            'Merged Dataframe', combined_df, 'body', "output/main_model_evaluation/data_cleaning/cleaned_data_frame.csv", data_cleaning)

        # Verifying the Cleaned Combine DataFrame
        # Concatenate the Cleaned DataFrame with the Merged DataFrame
        df_cleaned_combined = combine_columns_for_cleaning(
            combined_df, df_clean_body)

        verify_combined_dataframe(combined_df, df_cleaned_combined)

        # Save the cleaned combined DataFrame to CSV
        save_dataframe_to_csv(df_cleaned_combined,
                              file_paths['merged_cleaned_data_frame'])

        # ***************************** #
        #       Noise Injection         #
        # ***************************** #
        logging.info(f"Beginning Noise Injection...")
        noisy_df = generate_noisy_dataframe(
            df_cleaned_combined, 'output/main_model_evaluation/noise_injection/noisy_data_frame.csv')
        logging.info(f"Noise Injection completed.\n")

        # ************************* #
        #       Data Splitting      #
        # ************************* #
        logging.info(f"Beginning Data Splitting...")
        folds = stratified_k_fold_split(noisy_df)
        logging.info(f"Data Splitting completed.\n")

        # Initialize lists to store accuracies for each fold
        fold_train_accuracies = []
        fold_test_accuracies = []
        learning_curve_data = []

        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
            # ************************************************************ #
            #       Feature Extraction and Data Imbalance Handling         #
            # ************************************************************ #

            logging.info(
                f"Beginning Feature Extraction for Fold {fold_idx}...")

            # Define columns for categorical, numerical, and text data
            categorical_columns = ['sender', 'receiver']
            numerical_columns = ['https_count', 'http_count',
                                 'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
            text_column = 'cleaned_text'

            # Initialize BERT feature extractor and transformer
            bert_extractor = BERTFeatureExtractor()
            bert_transformer = BERTFeatureTransformer(
                feature_extractor=bert_extractor)

            # Define preprocessor for categorical and numerical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', Pipeline([
                        ('rare_cat_remover', RareCategoryRemover(
                            threshold=0.05)),  # Remove rare categories
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(
                            sparse_output=False, handle_unknown='ignore'))
                    ]), categorical_columns),
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), numerical_columns)
                ],
                remainder='passthrough'  # Keep other columns unchanged, like 'cleaned_text' and 'label'
            )

            # Define pipeline with preprocessor, BERT, and SMOTE
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('bert_features', bert_transformer),
                ('smote', SMOTE(random_state=42))
            ])

            # Call the function to either run the pipeline or load preprocessed data
            X_train_balanced, X_test_combined, y_train_balanced, y_test = run_pipeline_or_load(
                fold_idx=fold_idx,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                pipeline=pipeline,
                dir='output/main_model_evaluation/feature_extraction',
            )
            logging.info(
                f"Data for Fold {fold_idx} has been processed or loaded successfully.\n")

            # ***************************************** #
            #       Model Training and Evaluation       #
            # ***************************************** #
            logging.info(
                f"Beginning Model Training and Evaluation for Fold {fold_idx}...")
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "XGBoost": XGBClassifier(eval_metric='mlogloss'),
                "SVM": SVC(probability=True),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier(),
                "LightGBM": LGBMClassifier()
            }
            for model_name, model in models.items():
                logging.info(
                    f"Training {model_name} model for Fold {fold_idx}...")
                ensemble_model, test_accuracy = base_model_training(
                    X_train_balanced,
                    y_train_balanced,
                    X_test_combined,
                    y_test,
                    model,
                    model_name
                )
            fold_test_accuracies.append(test_accuracy)
            logging.info(
                f"Data for Fold {fold_idx} has been processed, model trained, and evaluated.\n")

        logging.info(f"Training and evaluation completed for all folds.\n")
        # Calculate and log the overall test accuracy
        mean_test_accuracy = np.mean(fold_test_accuracies)
        logging.info(f"Overall Test Accuracy: {mean_test_accuracy * 100:.2f}%")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Call the main function
if __name__ == "__main__":
    main()
