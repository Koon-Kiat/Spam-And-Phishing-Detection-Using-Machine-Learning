import os
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def stratified_k_fold_split(df, n_splits=3, random_state=42, output_dir='output/main_model_evaluation/data_splitting'):
    """
    Performs Stratified K-Fold splitting on the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    n_splits : int
        The number of splits for Stratified K-Fold.
    random_state : int
        The random state for reproducibility.
    output_dir : str
        The directory where the split data will be saved.

    Returns
    -------
    list
        A list of tuples containing train and test indices for each fold.
    """
    logging.info("Performing Stratified K-Fold splitting...")

    # Check if DataFrame contains necessary columns
    columns_to_use = ['sender', 'receiver', 'https_count', 'http_count',
                      'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']
    if not set(columns_to_use).issubset(df.columns):
        missing_cols = set(columns_to_use) - set(df.columns)
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Select columns to use for splitting
    df = df[columns_to_use]
    X = df.drop(columns=['label'])
    y = df['label']
    os.makedirs(output_dir, exist_ok=True)

    # Perform Stratified K-Fold splitting
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    folds = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        logging.info(f"Processing Fold {fold_idx}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Log the distribution of each split
        y_train_counts = y_train.value_counts().to_dict()
        y_test_counts = y_test.value_counts().to_dict()
        logging.info(
            f"Fold {fold_idx} - y_train distribution: {y_train_counts}, Total: {len(y_train)}")
        logging.info(
            f"Fold {fold_idx} - y_test distribution: {y_test_counts}, Total: {len(y_test)}")
        logging.info(
            f"Fold {fold_idx} - Total Combined: {len(y_test)+len(y_train)}")

        X_test_file = os.path.join(output_dir, f'X_Test_Fold{fold_idx}.csv')
        y_test_file = os.path.join(output_dir, f'y_Test_Fold{fold_idx}.csv')
        X_train_file = os.path.join(output_dir, f'X_Train_Fold{fold_idx}.csv')
        y_train_file = os.path.join(output_dir, f'y_Train_Fold{fold_idx}.csv')
        X_test.to_csv(X_test_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        X_train.to_csv(X_train_file, index=False)
        y_train.to_csv(y_train_file, index=False)
        folds.append((X_train, X_test, y_train, y_test))
        logging.info(f"Files saved to {output_dir}")
    logging.info("Completed Stratified K-Fold splitting.")

    return folds
