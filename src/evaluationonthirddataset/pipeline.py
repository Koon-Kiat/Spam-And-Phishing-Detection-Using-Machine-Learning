import os
import logging
import pickle
import numpy as np
import joblib


def save_output(data, labels, data_path, labels_path):
    np.savez(data_path, data=data)
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)


def load_output(data_path, labels_path):
    data = np.load(data_path)['data']
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return data, labels


def run_pipeline_or_load(data, labels, pipeline, dir):
    data_path = os.path.join(dir, 'processed_data.npz')
    labels_path = os.path.join(dir, 'processed_labels.pkl')
    preprocessor_path = os.path.join(dir, 'preprocessor.pkl')

    # Check if the files already exist
    if not all([os.path.exists(data_path), os.path.exists(labels_path), os.path.exists(preprocessor_path)]):
        logging.info("Running pipeline for the dataset...")

        # Process non-text features
        logging.info("Processing non-text features...")
        data_non_text = data.drop(columns=['cleaned_text'])

        # Fit the preprocessor
        logging.info("Fitting the preprocessor...")
        preprocessor = pipeline.named_steps['preprocessor']
        data_non_text_processed = preprocessor.fit_transform(data_non_text)
        feature_names = preprocessor.named_transformers_[
            'cat'].named_steps['encoder'].get_feature_names_out()
        logging.info(
            f"Columns in data after processing non-text features: {data_non_text_processed.shape}")
        if data_non_text_processed.shape[0] != labels.shape[0]:
            logging.error(
                f"Row mismatch: {data_non_text_processed.shape[0]} vs {labels.shape[0]}")
        logging.info("Non-text features processed.\n")

        # Save the preprocessor
        logging.info("Saving preprocessor...")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved preprocessor to {preprocessor_path}\n")

        # Transform the text features
        logging.info("Extracting BERT features for the dataset...")
        data_text_processed = pipeline.named_steps['bert_features'].transform(
            data['cleaned_text'].tolist())
        logging.info(
            f"Number of features extracted from BERT: {data_text_processed.shape}")
        logging.info("BERT features extracted.\n")

        # Combine processed features
        logging.info("Combining processed features...")
        data_combined = np.hstack(
            [data_non_text_processed, data_text_processed])
        logging.info(
            f"Total number of combined features: {data_combined.shape}")
        logging.info("Combined processed features.\n")

        logging.info("Applying PCA for dimensionality reduction...")
        data_combined = pipeline.named_steps['pca'].fit_transform(
            data_combined)

        # Log the number of features after PCA
        n_components = pipeline.named_steps['pca'].n_components_
        logging.info(f"Number of components after PCA: {n_components}")
        logging.info(f"Shape of data after PCA: {data_combined.shape}")

        # Save the preprocessed data
        logging.info("Saving processed data...")
        save_output(data_combined, labels, data_path, labels_path)
    else:
        # Load the preprocessor
        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)

        # Load the preprocessed data
        logging.info("Loading preprocessed data...")
        data_combined, labels = load_output(data_path, labels_path)

    return data_combined, labels

