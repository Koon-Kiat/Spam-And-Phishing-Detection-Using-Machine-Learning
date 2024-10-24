import os
import logging
import joblib
import numpy as np
import pickle
from collections import Counter


def run_pipeline_or_load(fold_idx, X_train, X_test, y_train, y_test, pipeline, dir):
    """
    Run the data processing pipeline for a specific fold in a stratified k-fold cross-validation.

    This code snippet performs the following tasks:
    1. Sets up the base directory and file paths for the fold.
    2. Checks if the preprocessed data files already exist.
    3. If the files do not exist:
        a. Logs the beginning of the pipeline for the fold.
        b. Processes non-text features.
        c. Fits the preprocessor and transforms the non-text features.
        d. Saves the preprocessor.
        e. Extracts BERT features for the text data.
        f. Combines the processed non-text and text features.
        g. Applies SMOTE to balance the training data.
        h. Applies PCA for dimensionality reduction.
        i. Logs the number of features after PCA.
        j. Saves the preprocessed data.
    4. If the files exist, loads the preprocessor and preprocessed data.
    5. Returns the balanced training data, combined test data, and their respective labels.

    Parameters
    ----------
    fold_idx : int
        The index of the current fold.
    X_train : pandas.DataFrame
        The training data.
    X_test : pandas.DataFrame
        The test data.
    y_train : pandas.Series
        The training labels.
    y_test : pandas.Series
        The test labels.
    pipeline : sklearn.pipeline.Pipeline
        The data processing pipeline.

    Returns
    -------
    tuple
        The balanced training data, combined test data, and their respective labels.
    """
    train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path = get_fold_paths(
        fold_idx, dir)

    # Check if the files already exist
    if not all([os.path.exists(train_data_path), os.path.exists(test_data_path), os.path.exists(train_labels_path), os.path.exists(test_labels_path), os.path.exists(preprocessor_path)]):
        logging.info(f"Running pipeline for fold {fold_idx}...")
        logging.info(f"Initial shape of X_train: {X_train.shape}")

        # Fit and transform the pipeline
        logging.info(f"Processing non-text features for fold {fold_idx}...")
        X_train_non_text = X_train.drop(columns=['cleaned_text'])
        X_test_non_text = X_test.drop(columns=['cleaned_text'])

        # Fit the preprocessor
        logging.info(f"Fitting the preprocessor for fold {fold_idx}...")
        preprocessor = pipeline.named_steps['preprocessor']
        X_train_non_text_processed = preprocessor.fit_transform(
            X_train_non_text)
        X_test_non_text_processed = preprocessor.transform(X_test_non_text)

        # Log feature names
        feature_names = preprocessor.named_transformers_[
            'cat'].named_steps['encoder'].get_feature_names_out()
        logging.info(
            f"Columns in X_train after processing non-text features: {X_train_non_text_processed.shape}")
        logging.info(f"Feature names: {feature_names}")

        # Check for consistent number of features
        if X_train_non_text_processed.shape[1] != X_test_non_text_processed.shape[1]:
            logging.error(f"Feature mismatch: {X_train_non_text_processed.shape[1]} vs {
                          X_test_non_text_processed.shape[1]}")

        if X_train_non_text_processed.shape[0] != y_train.shape[0]:
            logging.error(f"Row mismatch: {X_train_non_text_processed.shape[0]} vs {
                          y_train.shape[0]}")
        logging.info(f"Non text features processed for fold {fold_idx}.\n")

        # Save the preprocessor
        logging.info(f"Saving preprocessor for fold {fold_idx}...")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved preprocessor to {preprocessor_path}\n")

        # Transform the text features
        logging.info(f"Extracting BERT features for X_train for {fold_idx}...")
        X_train_text_processed = pipeline.named_steps['bert_features'].transform(
            X_train['cleaned_text'].tolist())
        logging.info(f"Extracting BERT features for X_test for {fold_idx}...")
        X_test_text_processed = pipeline.named_steps['bert_features'].transform(
            X_test['cleaned_text'].tolist())
        logging.info(f"Number of features extracted from BERT for fold {
                     fold_idx}: {X_train_text_processed.shape}")
        logging.info(f"Bert features extracted for fold {fold_idx}.\n")

        # Combine processed features
        logging.info(f"Combining processed features for fold {fold_idx}...")
        X_train_combined = np.hstack(
            [X_train_non_text_processed, X_train_text_processed])
        X_test_combined = np.hstack(
            [X_test_non_text_processed, X_test_text_processed])
        logging.info(f"Total number of combined features for fold {
                     fold_idx}: {X_train_combined.shape}")
        logging.info(f"Combined processed features for fold {fold_idx}.\n")

        logging.info(f"Class distribution before SMOTE for fold {
                     fold_idx}: {Counter(y_train)}")
        logging.info(
            f"Applying SMOTE to balance the training data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = pipeline.named_steps['smote'].fit_resample(
            X_train_combined, y_train)
        logging.info(f"Class distribution after SMOTE for fold {
                     fold_idx}: {Counter(y_train_balanced)}")
        logging.info(f"SMOTE applied for fold {fold_idx}.\n")

        '''
        logging.info(f"Applying PCA for dimensionality reduction for fold {fold_idx}...")
        X_train_balanced = pipeline.named_steps['pca'].fit_transform(X_train_balanced)
        X_test_combined = pipeline.named_steps['pca'].transform(X_test_combined)


        # Log the number of features after PCA
        n_components = pipeline.named_steps['pca'].n_components_
        logging.info(f"Number of components after PCA: {n_components}")
        logging.info(f"Shape of X_train after PCA: {X_train_balanced.shape}")
        '''
        logging.info(f"Shape of X_train: {X_train_balanced.shape}")

        # Save the preprocessed data
        logging.info(f"Saving processed data for fold {fold_idx}...")
        save_data_pipeline(X_train_balanced, y_train_balanced,
                           train_data_path, train_labels_path)
        save_data_pipeline(X_test_combined, y_test,
                           test_data_path, test_labels_path)
    else:
        # Load the preprocessor
        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)

        # Load the preprocessed data
        logging.info(f"Loading preprocessed data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = load_data_pipeline(
            train_data_path, train_labels_path)
        X_test_combined, y_test = load_data_pipeline(
            test_data_path, test_labels_path)

    return X_train_balanced, X_test_combined, y_train_balanced, y_test


def save_data_pipeline(data, labels, data_path, labels_path):
    """
    Save the data and labels to specified file paths.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be saved.
    labels : numpy.ndarray
        The labels to be saved.
    data_path : str
        The file path to save the data.
    labels_path : str
        The file path to save the labels.
    """
    np.savez(data_path, data=data)
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)


def load_data_pipeline(data_path, labels_path):
    """
    Load the data and labels from specified file paths.

    Parameters
    ----------
    data_path : str
        The file path to load the data from.
    labels_path : str
        The file path to load the labels from.

    Returns
    -------
    tuple
        A tuple containing the loaded data and labels.
    """
    data = np.load(data_path)['data']
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return data, labels


def get_fold_paths(fold_idx, base_dir='feature_extraction'):
    """
    Generates file paths for the train and test data and labels for the specified fold.

    Parameters
    ----------
    fold_idx : int
        The index of the fold.
    base_dir : str, optional
        The base directory where the data will be saved. Default is 'feature_extraction'.

    Returns
    -------
    tuple
        The file paths for the train data, test data, train labels, test labels, and preprocessor.
    """
    train_data_path = os.path.join(base_dir, f"Fold_{fold_idx}_Train_Data.npz")
    test_data_path = os.path.join(base_dir, f"Fold_{fold_idx}_Test_Data.npz")
    train_labels_path = os.path.join(
        base_dir, f"Fold_{fold_idx}_Train_Labels.pkl")
    test_labels_path = os.path.join(
        base_dir, f"Fold_{fold_idx}_Test_Labels.pkl")
    preprocessor_path = os.path.join(
        base_dir, f"Fold_{fold_idx}_Preprocessor.pkl")

    return train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path
