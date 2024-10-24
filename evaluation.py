import os  # Interact with the operating system
import json  # JSON parsing and manipulation
import codecs  # Codec registry and base classes
import re  # Regular expressions
import time  # Time-related functions
import logging  # Logging library
import warnings  # Warning control
import pickle  # Pickle (de)serialization
import string  # String operations
import random  # Random number generation


# Data Manipulation and Analysis
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import csv  # CSV file handling

# Data Visualization
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization
from wordcloud import WordCloud  # Generate word clouds

# Text Processing
import nltk  # Natural language processing
from nltk.corpus import stopwords  # Stop words
from nltk.corpus import wordnet  # WordNet corpus
from nltk.stem import WordNetLemmatizer  # Lemmatization
from nltk.tokenize import word_tokenize  # Tokenization
from nltk.data import find # Find NLTK resources
import contractions  # Expand contractions in text
import spacy  # NLP library


# Machine Learning Libraries
from sklearn.base import BaseEstimator, TransformerMixin  # Scikit-learn base classes
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA  # Principal Component Analysis
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                              GradientBoostingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS, 
                                             TfidfVectorizer, CountVectorizer)  # Text feature extraction
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)  # Metrics
from sklearn.model_selection import (GridSearchCV, train_test_split, 
                                       StratifiedKFold, cross_val_score, learning_curve)  # Model selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, 
                                   LabelEncoder, OrdinalEncoder, FunctionTransformer)  # Preprocessing
from sklearn.utils import resample  # Resampling utilities
from xgboost import XGBClassifier  # XGBoost Classifier
from sklearn.svm import SVC  # Support Vector Classifier

# Text Parsing and Email Handling
import email  # Email handling
import email.policy  # Email policies
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses

# Data Augmentation
from imblearn.over_sampling import SMOTE  # Handling imbalanced data
from transformers import MarianMTModel, MarianTokenizer  # Machine translation models

# Profiling and Job Management
import cProfile  # Profiling
from tqdm import tqdm  # Progress bar for loops
import joblib  # Job management
from joblib import Parallel, delayed  # Parallel processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # Concurrent processing


# Natural Language Toolkit (NLTK)
from collections import Counter  # Counter class for counting occurrences

# TensorFlow and PyTorch
import tensorflow as tf  # TensorFlow library
import torch  # PyTorch library
from torch.utils.data import DataLoader, Dataset  # Data handling in PyTorch

# Hyperparameter Optimization
import optuna  # Hyperparameter optimization
from optuna.samplers import TPESampler

# HTML and XML Parsing
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # HTML and XML parsing

# Spell Checking
from spellchecker import SpellChecker  # Spell checking

# Transformers Library
from transformers import (AdamW, BertForSequenceClassification, BertModel, 
                          BertTokenizer, Trainer, TrainingArguments)  # BERT models and training utilities
from transformers.utils import logging as transformers_logging

# Sparse Matrices
from scipy.sparse import csr_matrix, hstack  # Sparse matrix operations

# Typing Support
from typing import Dict, List, Union, Optional, Tuple  # Type hints
from tabulate import tabulate  # Pretty-print tabular data

# Datasets
from datasets import load_dataset  # Load datasets

from SpamAndPhishingDetection import (
    DatasetProcessor,
    log_label_percentages,
    count_urls,
    check_missing_values,
    load_or_clean_data,
    data_cleaning,
    TextProcessor,
    RareCategoryRemover
)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nlp = spacy.load('en_core_web_sm')  # Load the spaCy English model

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
tf.get_logger().setLevel('CRITICAL')  # Set TensorFlow logger to suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Configure the logging library to suppress TensorFlow logs

# Suppress warnings globally
warnings.simplefilter("ignore")  # Ignore all warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.keras')

# Optionally, configure transformers logging
transformers_logging.set_verbosity_error()

# Define loss function using the recommended method
loss_fn = tf.compat.v1.losses.sparse_softmax_cross_entropy



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
    """
    A class to extract email headers and other relevant information from email data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")



    def clean_links(self, links: List[str]) -> List[str]:
        """
        Clean the extracted links by removing unwanted characters and spaces.

        Parameters
        ----------
        links : List[str]
            The list of links to be cleaned.

        Returns
        -------
        List[str]
            The cleaned list of links.
        """
        cleaned_links = []
        for link in links:
            link = re.sub(r'[\'\[\]\s]+', '', link)
            link = re.sub(r'\\n+', ' ', link)
            link = link.strip()  # Trim leading and trailing spaces
            if link:  # Avoid appending empty links
                cleaned_links.append(link)

        return cleaned_links



    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        """
        Extract inline headers (From, To, Mail-To) from the email text.

        Parameters
        ----------
        email_text : str
            The email text to extract headers from.

        Returns
        -------
        Dict[str, Union[str, None]]
            A dictionary containing the extracted headers.
        """
        from_match = re.search(r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        from_header = from_match.group(1) if from_match else None
        to_header = to_match.group(1) if to_match else None
        mail_to_header = mail_to_match.group(1) if mail_to_match else None

        return {'From': from_header, 'To': to_header, 'Mail-To': mail_to_header}



    def extract_body_content(self, email_message: EmailMessage) -> str:
        """
        Extract the body content from an email message.

        Parameters
        ----------
        email_message : EmailMessage
            The email message to extract the body content from.

        Returns
        -------
        str
            The extracted body content.
        """
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
        """
        Count the occurrences of 'https' and 'http' in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        Dict[str, int]
            A dictionary containing the counts of 'https' and 'http'.
        """
        https_count = len(re.findall(r'https://', text))
        http_count = len(re.findall(r'http://', text))

        return {'https_count': https_count, 'http_count': http_count}



    def contains_blacklisted_keywords(self, text: str) -> int:
        """
        Count the occurrences of blacklisted keywords in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        int
            The count of blacklisted keywords in the text.
        """
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
        """
        Detect the number of URL shorteners in the list of links.

        Parameters
        ----------
        links : List[str]
            The list of links to check for URL shorteners.

        Returns
        -------
        int
            The count of URL shorteners in the list of links.
        """
        shortener_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 
            'adf.ly', 'bl.ink', 'lnkd.in', 'shorte.st', 'mcaf.ee', 'q.gs', 'po.st', 
            'bc.vc', 's.coop', 'u.to', 'cutt.ly', 't2mio.com', 'rb.gy', 'clck.ru', 
            'shorturl.at', '1url.com', 'hyperurl.co', 'urlzs.com', 'v.gd', 'x.co'
        ]
        short_urls = [link for link in links if any(domain in link for domain in shortener_domains)]
        return len(short_urls)



    def count_ip_addresses(self, text: str) -> int:
        """
        Count the occurrences of IP addresses in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        int
            The count of IP addresses in the text.
        """
        ip_pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'

        return len(re.findall(ip_pattern, text))
    


    def extract_header_evaluation(self) -> pd.DataFrame:
        """
        Extract headers and other relevant information from the email data for SpamAssassin dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted headers and information.
        """
        headers_list: List[Dict[str, Union[str, List[str], int]]] = []
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                email_message = BytesParser(policy=policy.default).parsebytes(email_text.encode('utf-8'))
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get('Mail-To') if email_message.get('Mail-To') else None

                if not from_header or not to_header:
                    inline_headers = self.extract_inline_headers(email_text)
                    from_header = inline_headers['From'] or from_header
                    to_header = inline_headers['To'] or to_header
                    mail_to_header = inline_headers['Mail-To'] or mail_to_header

                from_header = from_header if from_header else None
                to_header = to_header if to_header else None
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
                has_ip_address = self.count_ip_addresses(body_content)



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
                headers_list.append(
                    {'sender': None, 'receiver': None, 'mailto': None, 'texturls': [], 'blacklisted_keywords_count': 0, 'short_urls': [], 'has_ip_address': 0})
        self.headers_df = pd.DataFrame(headers_list)

        return self.headers_df
    


    def save_to_csv(self, file_path: str):
        """
        Save the extracted headers DataFrame to a CSV file.

        Parameters
        ----------
        file_path : str
            The path to save the CSV file.

        Raises
        ------
        ValueError
            If no header information has been extracted.
        """
        if not self.headers_df.empty:
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError(
                "No header information extracted. Please run extract_headers() first.")



def save_data_pipeline(data, labels, data_path, labels_path):
    np.savez(data_path, data=data)
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)



def load_data_pipeline(data_path, labels_path):
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
        feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
        logging.info(f"Columns in data after processing non-text features: {data_non_text_processed.shape}")
        if data_non_text_processed.shape[0] != labels.shape[0]:
            logging.error(f"Row mismatch: {data_non_text_processed.shape[0]} vs {labels.shape[0]}")
        logging.info("Non-text features processed.\n")

        # Save the preprocessor
        logging.info("Saving preprocessor...")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved preprocessor to {preprocessor_path}\n")

        # Transform the text features
        logging.info("Extracting BERT features for the dataset...")
        data_text_processed = pipeline.named_steps['bert_features'].transform(data['cleaned_text'].tolist())
        logging.info(f"Number of features extracted from BERT: {data_text_processed.shape}")
        logging.info("BERT features extracted.\n")

        # Combine processed features
        logging.info("Combining processed features...")
        data_combined = np.hstack([data_non_text_processed, data_text_processed])
        logging.info(f"Total number of combined features: {data_combined.shape}")
        logging.info("Combined processed features.\n")

        logging.info("Applying PCA for dimensionality reduction...")
        data_combined = pipeline.named_steps['pca'].fit_transform(data_combined)

        # Log the number of features after PCA
        n_components = pipeline.named_steps['pca'].n_components_
        logging.info(f"Number of components after PCA: {n_components}")
        logging.info(f"Shape of data after PCA: {data_combined.shape}")

        # Save the preprocessed data
        logging.info("Saving processed data...")
        save_data_pipeline(data_combined, labels, data_path, labels_path)
    else:
        # Load the preprocessor
        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)

        # Load the preprocessed data
        logging.info("Loading preprocessed data...")
        data_combined, labels = load_data_pipeline(data_path, labels_path)

    return data_combined, labels



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
    


def stratified_k_fold_split(df, output_dir, n_splits=3, random_state=42):
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


        X_test_file = os.path.join(output_dir, f'X_Test_Fold{fold_idx}.csv')
        y_test_file = os.path.join(output_dir, f'y_Test_Fold{fold_idx}.csv')
        X_train_file = os.path.join(output_dir, f'X_Train_Fold{fold_idx}.csv')
        y_train_file = os.path.join(output_dir, f'y_Train_Fold{fold_idx}.csv')
        X_test.to_csv(X_test_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        X_train.to_csv(X_train_file, index=False)
        y_train.to_csv(y_train_file, index=False)
        folds.append((X_train, X_test, y_train, y_test))
    logging.info("Completed Stratified K-Fold splitting.")

    return folds



logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'Datasets', 'Phishing_Email.csv')
    PreprocessedEvaluationDataset = os.path.join(base_dir, 'Dataset Evaluation', 'Data Preprocessing', 'Preprocessed_Evaluation_Dataset.csv')
    ExtractedEvaluationHeaderFile = os.path.join(base_dir, 'Dataset Evaluation', 'Feature Engineering', 'Extracted_Evaluation_HeaderFile.csv')
    CleanedEvaluationDataFrame = os.path.join(base_dir, 'Dataset Evaluation', 'Data Cleaning', 'Cleaned_Evaluation_DataFrame.csv')
    MergedEvaluationFile = os.path.join(base_dir, 'Dataset Evaluation', 'Data Integration', 'Merged_Evaluation.csv')
    MergedCleanedDataFrame = os.path.join(base_dir, 'Dataset Evaluation', 'Data Cleaning', 'Merged_Cleaned_DataFrame.csv')


    df_evaluation = pd.read_csv(dataset)

    # ****************************** #
    #       Data Preprocessing       #
    # ****************************** #
    logging.info("Beginning Data Preprocessing...")

    # Rename 'Email Type' column to 'Label' and map the values
    df_evaluation['label'] = df_evaluation['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df_evaluation = df_evaluation.rename(columns={'Email Text': 'text'})

    # Drop the original 'Email Type' column if no longer needed
    df_evaluation = df_evaluation.drop(columns=['Email Type'])

    processor_evaluation = DatasetProcessor(df_evaluation, "text", "Evaluation Dataset", PreprocessedEvaluationDataset)
    df_processed_evaluation = processor_evaluation.process_dataset()

    log_label_percentages(df_processed_evaluation, 'Evaluation Dataset')
    logging.info("Data Preprocessing completed.\n")


    # ****************************** #
    #       Feature Engineering      #
    # ****************************** #
    logging.info("Beginning Feature Engineering...")
    evaluation_email_headers = load_or_extract_headers(df_processed_evaluation, ExtractedEvaluationHeaderFile, EmailHeaderExtractor, 'Evaluation')
    logging.info("Email header extraction and saving from Evaluation completed.")
    evaluation_email_headers['urls'] = evaluation_email_headers['texturls'].apply(count_urls)
    evaluation_email_headers.drop(columns=['mailto'], inplace=True) # Drop the 'mailto' column
    evaluation_email_headers.drop(columns=['texturls'], inplace=True) # Drop the 'texturls' column
    logging.info("Feature Engineering completed.\n")
   


    # ****************************** #
    #       Data Integration         #
    # ****************************** #
    logging.info("Beginning Data Integration...")
    df_processed_evaluation.reset_index(inplace=True)
    evaluation_email_headers.reset_index(inplace=True)
    #evaluation_email_headers.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
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
    logging.info("Data Integration completed.\n")
        

    # ************************* #
    #       Data Cleaning       #
    # ************************* #
    logging.info("Beginning Data Cleaning...")
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
    


    # ***************************** #
    #       Model Prediction        #
    # ***************************** #
    categorical_columns = ['sender', 'receiver']
    numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
    text_column = 'cleaned_text'

    for col in numerical_columns:
        df_evaluation_clean_combined[col] = pd.to_numeric(df_evaluation_clean_combined[col], errors='coerce').fillna(0)


    # Initialize BERT feature extractor and transformer
    bert_extractor = BERTFeatureExtractor()
    bert_transformer = BERTFeatureTransformer(feature_extractor=bert_extractor)


    # Define preprocessor for categorical and numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                #('rare_cat_remover', RareCategoryRemover(threshold=0.05)),  # Remove rare categories
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), categorical_columns),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values
                ('scaler', StandardScaler())
            ]), numerical_columns)
        ],
        remainder='passthrough'  # Keep other columns unchanged, like 'cleaned_text' and 'label'
    )

    # Define pipeline with preprocessor, BERT, SMOTE, and PCA
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('bert_features', bert_transformer),  # Custom transformer for BERT
        #('smote', SMOTE(random_state=42)),  # Apply SMOTE after feature extraction
        ('pca', PCA(n_components=777))
    ])

    data_dir = os.path.join(base_dir, 'Dataset Evaluation', 'Feature Extraction')  # Directory where you want to save the processed data

    # Assuming df_evaluation_clean_combined is the DataFrame you are working with
    X = df_evaluation_clean_combined.drop(columns=['label'])
    y = df_evaluation_clean_combined['label']

    # Run the pipeline or load processed data
    processed_data, processed_labels = run_pipeline_or_load(X, y, pipeline, data_dir)
    


    # Load the saved model
    Main_Model = os.path.join(base_dir, 'Models & Parameters')
    Base_Model_No_Optuna = os.path.join(base_dir, 'Test Models', 'Base Models (No Optuna)')
    Base_Model_Optuna = os.path.join(base_dir, 'Test Models', 'Base Models (Optuna)')
    Stacked_Model_Optuna = os.path.join(base_dir, 'Test Models', 'Stacked Models (Optuna)')

    # List of folders containing models
    model_folders = [Main_Model, Base_Model_No_Optuna, Base_Model_Optuna, Stacked_Model_Optuna]

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
                    classification = classification_report(processed_labels, y_pred, target_names=['Safe', 'Not Safe'], zero_division=1)
                    results.append({
                        'Model': model_name,
                        'Folder': folder_name,
                        'Accuracy': f"{test_accuracy * 100:.2f}%",
                        'Confusion Matrix': confusion,
                        'Classification Report': classification
                    })
                except Exception as e:
                    logging.error(f"Error loading or evaluating model {model_file}: {e}")
            else:
                print(f"Skipping invalid path: {model_path}")

    # Convert results to a DataFrame for better formatting
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    output_path = os.path.join('Dataset Evaluation', 'Model_Evaluation_Result.csv')
    results_df[['Model', 'Folder', 'Accuracy']].to_csv(output_path, index=False)

    # Print the results in a table format
    print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))


if __name__ == '__main__':
    main()
