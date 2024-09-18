import numpy as np  # Numerical operations
import pandas as pd 
import os
import logging
from SpamAndPhishingDetection import DatasetProcessor
from SpamAndPhishingDetection import log_label_percentages
from SpamAndPhishingDetection import count_urls
import re
from typing import Optional, List, Dict, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed  # Multithreading
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from functools import lru_cache  # Least Recently Used (LRU) cache
from tqdm import tqdm  # Progress bar
from email.utils import parseaddr, getaddresses
from email.policy import default
from SpamAndPhishingDetection import check_missing_values
from SpamAndPhishingDetection import load_or_clean_data
from SpamAndPhishingDetection import data_cleaning
from SpamAndPhishingDetection import TextProcessor
from email.utils import formataddr
from email.headerregistry import Address
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def load_or_extract_headers(df: pd.DataFrame, file_path: str, extractor_class, dataset_type: str) -> pd.DataFrame:
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
            logging.info(f"File {file_path} already exists. Loading from file.")

            return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist. Extracting headers for dataset: {dataset_type}.")
        header_extractor = extractor_class(df)
        
        # Check dataset type and call the corresponding extraction function
        if dataset_type == "Evaluation":
            headers_df = header_extractor.extract_header_evaluation()
        else:
            raise ValueError(f"Dataset type {dataset_type} not supported. Please use 'Evaluation'.")
        header_extractor.save_to_csv(file_path)
        logging.info(f"Email header extraction and saving to {file_path} completed for dataset: {dataset_type}.")
        
        return headers_df
 


class EmailHeaderExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")

    def clean_links(self, links: List[str]) -> List[str]:
        cleaned_links = []
        for link in links:
            link = re.sub(r'[\'\[\]\s]+', '', link)
            link = re.sub(r'\\n+', ' ', link)
            link = link.strip()  # Trim leading and trailing spaces
            if link:  # Avoid appending empty links
                cleaned_links.append(link)
        return cleaned_links

    def extract_inline_headers(self, email_text: str) -> Dict[str, Optional[str]]:
        from_match = re.search(r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        from_header = from_match.group(1) if from_match else None
        to_header = to_match.group(1) if to_match else None
        mail_to_header = mail_to_match.group(1) if mail_to_match else None

        return {'From': from_header, 'To': to_header, 'Mail-To': mail_to_header}

    def extract_body_content(self, email_message) -> str:
        body_content = ""
        if email_message.is_multipart():
            for part in email_message.iter_parts():
                if part.get_content_type() == 'text/plain':
                    body_content += part.get_payload(decode=True).decode(errors='ignore')
                elif part.get_content_type() == 'text/html':
                    body_content += part.get_payload(decode=True).decode(errors='ignore')
        else:
            body_content = email_message.get_payload(decode=True).decode(errors='ignore')
        return body_content

    def count_https_http(self, text: str) -> Dict[str, int]:
        https_count = len(re.findall(r'https://', text))
        http_count = len(re.findall(r'http://', text))
        return {'https_count': https_count, 'http_count': http_count}

    def contains_blacklisted_keywords(self, text: str) -> int:
        blacklisted_keywords = [
            'click now', 'verify now', 'urgent', 'free', 'winner',
            'limited time', 'act now', 'your account', 'risk', 'account update',
            'important update', 'security alert', 'confirm your identity',
            'password reset', 'access your account', 'log in', 'claim your prize',
            'congratulations', 'update required', 'you have been selected',
            'validate your account', 'final notice', 'click here', 'confirm now',
            'take action', 'unauthorized activity', 'sign in', 'redeem now',
            'you are a winner', 'download now', 'urgent action required',
            'reset password', 'limited offer', 'exclusive deal', 'verify account',
            'bank account', 'payment declined', 'upgrade required', 'respond immediately'
        ]
        keyword_count = 0
        for keyword in blacklisted_keywords:
            keyword_count += len(re.findall(re.escape(keyword), text, re.IGNORECASE))
        return keyword_count

    def detect_url_shorteners(self, links: List[str]) -> int:
        shortener_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 
            'adf.ly', 'bl.ink', 'lnkd.in', 'shorte.st', 'mcaf.ee', 'q.gs', 'po.st', 
            'bc.vc', 's.coop', 'u.to', 'cutt.ly', 't2mio.com', 'rb.gy', 'clck.ru', 
            'shorturl.at', '1url.com', 'hyperurl.co', 'urlzs.com', 'v.gd', 'x.co'
        ]
        short_urls = [link for link in links if any(domain in link for domain in shortener_domains)]
        return len(short_urls)

    def contains_ip_address(self, text: str) -> bool:
        ip_pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'
        return bool(re.search(ip_pattern, text))

    

    def extract_header_evaluation(self) -> pd.DataFrame:
        headers_list: List[Dict[str, Union[str, List[str], int]]] = []

        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                email_message = BytesParser(policy=policy.default).parsebytes(email_text.encode('utf-8'))

                # Extract headers
                from_header = email_message.get('From', '')
                to_header = email_message.get('To', '')
                mail_to_header = email_message.get('Mail-To', '')

                # Extract 'From' and 'To' headers with proper handling for groups
                from_header = self.extract_single_address(from_header)
                to_header = self.extract_single_address(to_header)

                mail_to_header = mail_to_header if mail_to_header else None
                body_content = self.extract_body_content(email_message)
                logging.debug(f"Email body content: {body_content}")

                # Extract URLs from body content
                url_pattern = r'https?:\/\/[^\s\'"()<>]+'
                links = re.findall(url_pattern, body_content)
                links = self.clean_links(links)

                # Count blacklisted keywords, http/https, short URLs, and IP addresses in the email body
                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.contains_blacklisted_keywords(body_content)
                short_urls = self.detect_url_shorteners(links)
                has_ip_address = self.contains_ip_address(body_content)

                headers_list.append({
                    'sender': from_header,
                    'receiver': to_header,
                    'mailto': mail_to_header,
                    'texturls': links,
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address
                })

            except Exception as e:
                logging.error(f"Error parsing email: {e}")
                logging.debug(f"Email text: {email_text}")
                headers_list.append({
                    'sender': None,
                    'receiver': None,
                    'mailto': None,
                    'texturls': [],
                    'https_count': 0,
                    'http_count': 0,
                    'blacklisted_keywords_count': 0,
                    'short_urls': [],
                    'has_ip_address': False
                })

        self.headers_df = pd.DataFrame(headers_list)
        return self.headers_df
    

    def extract_single_address(self, header_value: Optional[Union[str, List[str], Tuple[str, str]]]) -> Optional[str]:
        """
        Extracts a single email address from the header value.
        Handles cases where the header value may be a string, list, tuple, or Group object.
        """
        logging.debug(f"Processing header value: {header_value}")
        
        # Handle string header value
        if isinstance(header_value, str):
            addresses = getaddresses([header_value])
            for _, email in addresses:
                if email:
                    return email
        
        # Handle list or tuple of header values
        elif isinstance(header_value, (list, tuple)):
            for item in header_value:
                if isinstance(item, str):
                    addresses = getaddresses([item])
                    for _, email in addresses:
                        if email:
                            return email
                elif isinstance(item, tuple):
                    # Handle cases where item might be a tuple of (name, email)
                    if len(item) == 2 and isinstance(item[1], str):
                        return item[1]
        
        # Handle Group object (or other complex structures)
        elif hasattr(header_value, 'addresses'):
            addresses = getaddresses([str(header_value)])
            for _, email in addresses:
                if email:
                    return email
        else:
            logging.error(f"Unexpected type for header_value: {type(header_value)}")
            logging.debug(f"Value: {header_value}")

        return None



    def save_to_csv(self, file_path: str):
        if not self.headers_df.empty:
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError("No header information extracted. Please run extract_header_evaluation() first.")



def get_fold_paths(fold_idx, base_dir='Evaluation'):
    train_data_path = os.path.join(base_dir, f"fold_{fold_idx}_train_data.npz")
    test_data_path = os.path.join(base_dir, f"fold_{fold_idx}_test_data.npz")
    train_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_train_labels.pkl")
    test_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_test_labels.pkl")
    preprocessor_path = os.path.join(base_dir, f"fold_{fold_idx}_preprocessor.pkl")
    
    return train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path



def save_data_pipeline(data, labels, data_path, labels_path):
    np.savez(data_path, data=data)
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)



def load_data_pipeline(data_path, labels_path):
    data = np.load(data_path)['data']
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return data, labels



def run_pipeline_or_load(fold_idx, X_train, X_test, y_train, y_test, pipeline):
    # Define paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Evaluation')
    os.makedirs(base_dir, exist_ok=True)
    train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path = get_fold_paths(fold_idx, base_dir)

    # Check if the files already exist
    if not all([os.path.exists(train_data_path), os.path.exists(test_data_path), os.path.exists(train_labels_path), os.path.exists(test_labels_path), os.path.exists(preprocessor_path)]):
        logging.info(f"Running pipeline for fold {fold_idx}...")

        # Fit and transform the pipeline
        logging.info(f"Processing non-text features for fold {fold_idx}...")
        X_train_non_text = X_train.drop(columns=['cleaned_text'])
        X_test_non_text = X_test.drop(columns=['cleaned_text'])

        # Fit the preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        X_train_non_text_processed = preprocessor.fit_transform(X_train_non_text)
        X_test_non_text_processed = preprocessor.transform(X_test_non_text)

        # Save the preprocessor
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved preprocessor to {preprocessor_path}")

        # Transform the text features
        logging.info(f"Extracting BERT features for X_train for {fold_idx}...")
        X_train_text_processed = pipeline.named_steps['bert_features'].transform(X_train['cleaned_text'].tolist())
        logging.info(f"Extracting BERT features for X_test for {fold_idx}...")
        X_test_text_processed = pipeline.named_steps['bert_features'].transform(X_test['cleaned_text'].tolist())

        # Combine processed features
        logging.info(f"Combining processed features for fold {fold_idx}...")
        X_train_combined = np.hstack([X_train_non_text_processed, X_train_text_processed])
        X_test_combined = np.hstack([X_test_non_text_processed, X_test_text_processed])

        # Apply SMOTE
        logging.info(f"Applying SMOTE to balance the training data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = pipeline.named_steps['smote'].fit_resample(X_train_combined, y_train)

        # Save the preprocessed data
        logging.info(f"Saving processed data for fold {fold_idx}...")
        save_data_pipeline(X_train_balanced, y_train_balanced, train_data_path, train_labels_path)
        save_data_pipeline(X_test_combined, y_test, test_data_path, test_labels_path)
    else:
        # Load the preprocessor
        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)

        # Load the preprocessed data
        logging.info(f"Loading preprocessed data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = load_data_pipeline(train_data_path, train_labels_path)
        X_test_combined, y_test = load_data_pipeline(test_data_path, test_labels_path)

    return X_train_balanced, X_test_combined, y_train_balanced, y_test


class BERTFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(self.feature_extractor.extract_features(X))
    


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length



    def __len__(self):
        return len(self.texts)



    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Ensure that the dimensions are correctly handled
        input_ids = inputs['input_ids'].squeeze(dim=0)
        attention_mask = inputs['attention_mask'].squeeze(dim=0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }



class BERTFeatureExtractor:
    def __init__(self, max_length=128, device=None):
        logging.info("Initializing BERT Feature Extractor...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Ensure model is on the right device



    def extract_features(self, texts, batch_size=16):
        if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input should be a list or tuple of strings.")
        features = []
        self.model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT features", leave=True):
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move back to CPU
                features.extend(batch_features)

        return features
    

    # Redundant function
    def save_features(self, features, features_path):
        logging.info(f"Saving features to {features_path}.")
        np.save(features_path, features)
        logging.info(f"Features saved to {features_path}.")


    # Redundant function
    def load_features(self, features_path):
        logging.info(f"Loading features from {features_path}.")
        if os.path.exists(features_path):
            logging.info(f"Loading features from {features_path}.")
            return np.load(features_path)
        else:
            logging.info("Features file not found. Extracting features.")

            return None


def smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced


def stratified_k_fold_split(df, n_splits=3, random_state=42, output_dir='Evaluation'):
    logging.info("Performing Stratified K-Fold splitting...")
    

    # Check if DataFrame contains necessary columns
    columns_to_use = ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']
    if not set(columns_to_use).issubset(df.columns):
        missing_cols = set(columns_to_use) - set(df.columns)
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    

    # Select columns to use for splitting
    df = df[columns_to_use]
    X = df.drop(columns=['label'])
    y = df['label']
    os.makedirs(output_dir, exist_ok=True)


    # Perform Stratified K-Fold splitting
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        logging.info(f"Processing Fold {fold_idx}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        # Log the distribution of each split
        y_train_counts = y_train.value_counts().to_dict()
        y_test_counts = y_test.value_counts().to_dict()
        logging.info(f"Fold {fold_idx} - y_train distribution: {y_train_counts}, Total: {len(y_train)}")
        logging.info(f"Fold {fold_idx} - y_test distribution: {y_test_counts}, Total: {len(y_test)}")
        logging.info(f"Fold {fold_idx} - Total Combined: {len(y_test)+len(y_train)}")


        X_test_file = os.path.join(output_dir, f'X_test_fold{fold_idx}.csv')
        y_test_file = os.path.join(output_dir, f'y_test_fold{fold_idx}.csv')
        X_test.to_csv(X_test_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        folds.append((X_train, X_test, y_train, y_test))
    logging.info("Completed Stratified K-Fold splitting.")

    return folds


logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)


def main():
    # ****************************** #
    #       Data Preprocessing       #
    # ****************************** #
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'Datasets', 'Phishing_Email.csv')
    ExtractedEvaluationHeaderFile = os.path.join(base_dir, 'Evaluation', 'ExtractedEvaluationHeaderFile.csv')
    CleanedEvaluationDataFrame = os.path.join(base_dir, 'Evaluation', 'CleanedEvaluationDataFrame.csv')
    MergedEvaluationFile = os.path.join(base_dir, 'Evaluation', 'MergedEvaluation.csv')
    MergedCleanedDataFrame = os.path.join(base_dir, 'Evaluation', 'MergedCleanedDataFrame.csv')
    SavedModel = os.path.join(base_dir, 'Models & Parameters', 'ensemble_model_fold_1.pkl')
    SavedPreprocessor = os.path.join(base_dir, 'Models & Parameters', 'fold_1_preprocessor.pkl')
    SavedBertTransformer = os.path.join(base_dir, 'Models & Parameters', 'fold_1_bert_transformer.pkl')

    df_evaluation = pd.read_csv(dataset)  

    # Rename 'Email Type' column to 'Label' and map the values
    df_evaluation['label'] = df_evaluation['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df_evaluation = df_evaluation.rename(columns={'Email Text': 'text'})

    # Drop the original 'Email Type' column if no longer needed
    df_evaluation = df_evaluation.drop(columns=['Email Type'])

    processor_evaluation = DatasetProcessor(df_evaluation, "text", "Evaluation Dataset")
    df_processed_evaluation = processor_evaluation.process_dataset()

    log_label_percentages(df_processed_evaluation, 'Evaluation Dataset')


    # ****************************** #
    #       Feature Engineering      #
    # ****************************** #
    evaluation_email_headers = load_or_extract_headers(df_processed_evaluation, ExtractedEvaluationHeaderFile, EmailHeaderExtractor, 'Evaluation')
    logging.info("Email header extraction and saving from Spam Assassin completed.\n")
    evaluation_email_headers['urls'] = evaluation_email_headers['texturls'].apply(count_urls)
    evaluation_email_headers.drop(columns=['mailto'], inplace=True) # Drop the 'mailto' column
    evaluation_email_headers.drop(columns=['texturls'], inplace=True) # Drop the 'texturls' column
   


    # ****************************** #
    #       Data Integration         #
    # ****************************** #
    df_processed_evaluation.reset_index(inplace=True)
    evaluation_email_headers.reset_index(inplace=True)
    evaluation_email_headers.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
    if len(df_processed_evaluation) == len(evaluation_email_headers):
        # Merge dataframes
        merged_evaluation = pd.merge(df_processed_evaluation, evaluation_email_headers, on='index', how='left')
        # Rename and reorder columns
        merged_evaluation = merged_evaluation.rename(columns={'text': 'body'})
        merged_evaluation = merged_evaluation[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']]
        
        # Log missing rows
        missing_in_merged = merged_evaluation[merged_evaluation['index'].isnull()]
        logging.info(f"Number of missing rows in Evaluation Dataframe: {len(missing_in_merged)}")
        logging.info(f'Total rows in Processed Evaluation Dataframe: {len(df_processed_evaluation)}')
        logging.info(f"Total rows in Evaluation Dataframe: {len(merged_evaluation)}")
        
        # Drop index column
        merged_evaluation.drop(columns=['index'], inplace=True)                         
    else:
        logging.error("Length of the two dataframes do not match. Please check the dataframes.")
        raise ValueError("Length of the two dataframes do not match. Please check the dataframes.")
    
    if len(merged_evaluation) != len(df_processed_evaluation):
        logging.error("The number of rows in the Merge Evaluation Dataframe does not match the Processed Evaluation Dataframe.")
        raise ValueError("The number of rows in the Merge Evaluation Dataframe does not match the Processed Evaluation Dataframe.")
    else:
        logging.info("The number of rows in the Merge Evaluation Dataframe matches the Processed Evaluation Dataframe.")
        merged_evaluation.to_csv(MergedEvaluationFile, index=False)
        logging.info(f"Data successfully saved to: {MergedEvaluationFile}")
        

    # ************************* #
    #       Data Cleaning       #
    # ************************* #
    df_evaluation_clean = load_or_clean_data("Evaluation", merged_evaluation, 'body', CleanedEvaluationDataFrame, data_cleaning)
    logging.info (f"Data Cleaning completed.\n")


    merged_evaluation_reset = merged_evaluation.reset_index(drop=True)
    df_evaluation_clean_reset = df_evaluation_clean.reset_index(drop=True)
    df_evaluation_clean_combined = pd.concat([
            merged_evaluation_reset[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'label']],  # Select necessary columns from merged
            df_evaluation_clean_reset[['cleaned_text']]  # Select the cleaned_text and label from df_clean
        ], axis=1)

    merged_evaluation_labels = merged_evaluation['label'].unique()
    df_evaluation_clean_combined_labels = df_evaluation_clean_combined['label'].unique()
    if set(merged_evaluation_labels) != set(df_evaluation_clean_combined_labels):
        logging.error(f"Labels in Combined DataFrame do not match those in Cleaned Combined DataFrame. "
                    f"Combined DataFrame labels: {merged_evaluation_labels}, "
                    f"Cleaned Combined DataFrame labels: {df_evaluation_clean_combined_labels}")
        raise ValueError("Labels do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info("Labels in Combined DataFrame match those in Cleaned Combined DataFrame.")
    merged_evaluation_label_counts = merged_evaluation['label'].value_counts().sort_index()
    df_evaluation_clean_combined_counts = df_evaluation_clean_combined['label'].value_counts().sort_index()
    if not merged_evaluation_label_counts.equals(df_evaluation_clean_combined_counts):
        logging.error("Label distributions in Combined DataFrame do not match those in Cleaned Combined DataFrame.")
        logging.error(f"Combined DataFrame distributions:\n{merged_evaluation_label_counts}")
        logging.error(f"Cleaned Combined DataFrame distributions:\n{df_evaluation_clean_combined_counts}")
        raise ValueError("Label distributions do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info("Label distributions in Combined DataFrame match those in Cleaned Combined DataFrame.")


        # Final columns to keep
        df_evaluation_clean_combined = df_evaluation_clean_combined[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']]
        logging.info(f"Final combined DataFrame has {len(df_evaluation_clean_combined)} rows and columns: {df_evaluation_clean_combined.columns.tolist()}")
        df_evaluation_clean_combined.to_csv(MergedCleanedDataFrame, index=False)
        logging.info(f"Data Cleaning completed.\n")
        numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls']
        for col in numerical_columns:
            df_evaluation_clean_combined[col] = pd.to_numeric(df_evaluation_clean_combined[col], errors='coerce').fillna(0)
    
    # ************************* #
    #       Data Splitting      #
    # ************************* #

    logging.info(f"Beginning Data Splitting...")
    folds = stratified_k_fold_split(df_evaluation_clean_combined)
    logging.info(f"Data Splitting completed.\n")

    # Initialize lists to store accuracies for each fold
    fold_train_accuracies = []
    fold_test_accuracies = []

    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
        categorical_columns = ['sender', 'receiver', 'has_ip_address']
        numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls']
        text_column = 'cleaned_text'


        # Initialize BERT feature extractor and transformer
        bert_extractor = BERTFeatureExtractor()
        bert_transformer = BERTFeatureTransformer(feature_extractor=bert_extractor)


        # Define preprocessor for categorical and numerical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with the most frequent
                    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]), categorical_columns),
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values with the mean
                    ('scaler', StandardScaler())
                ]), numerical_columns)
            ],
            remainder='passthrough'  # Keep other columns unchanged, like 'cleaned_text' and 'label'
        )


        # Define pipeline with preprocessor, BERT, and SMOTE
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('bert_features', bert_transformer),  # Custom transformer for BERT
            ('smote', SMOTE(random_state=42))  # Apply SMOTE after feature extraction
        ])
        # Call the function to either run the pipeline or load preprocessed data
        X_train_balanced, X_test_combined, y_train_balanced, y_test = run_pipeline_or_load(
            fold_idx=fold_idx,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pipeline=pipeline
        )
        logging.info(f"Data for Fold {fold_idx} has been processed or loaded successfully.")

        saved_model = joblib.load(SavedModel)
        X_test_transformed = saved_model.transform(X_test_combined)

        # Make predictions on the test data
        y_pred = saved_model.predict(X_test_transformed)

        # Evaluate the model
        test_accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        logging.info(f"Test Accuracy: {test_accuracy}")
        logging.info(f"Classification Report:\n{classification_rep}")

         # Load the saved model
        saved_model = joblib.load(SavedModel)

        # Make predictions on the test data
        y_pred = saved_model.predict(X_test_combined)

        # Evaluate the model
        test_accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        logging.info(f"Fold {fold_idx} - Test Accuracy: {test_accuracy}")
        logging.info(f"Fold {fold_idx} - Classification Report:\n{classification_rep}")

        fold_test_accuracies.append(test_accuracy)





if __name__ == '__main__':
    main()