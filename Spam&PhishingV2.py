# Description: This file is used to test the data cleaning and processing functions.

# Data manipulation
import codecs  # Codec registry and base classes
import cProfile  # Profiling

# Email parsing
import email  # Email handling
import email.policy  # Email policies
import json  # JSON parsing and manipulation

# Logging
import logging  # Logging library

# Operating system interfaces
import os  # Interact with the operating system
import re  # Regular expressions

# String and regular expression operations
import string  # String operations

# Profiling and job management
import time  # Time-related functions
import urllib.parse  # URL parsing

# Warnings
import warnings  # Warning control

# Concurrent execution
from concurrent.futures import ThreadPoolExecutor, as_completed  # Multithreading
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from functools import lru_cache  # Least Recently Used (LRU) cache

# Typing support
from typing import Dict, List, Union  # Type hints
from unittest.mock import patch

# Text processing
import contractions  # Expand contractions in text
import joblib  # Job management
from joblib import Parallel, delayed  # Parallel processing

# Data visualization
import matplotlib.pyplot as plt  # Plotting library

# Natural Language Toolkit (NLTK)
import nltk  # Natural language processing
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import seaborn as sns  # Statistical data visualization

# TensorFlow
import tensorflow as tf  # TensorFlow library

# PyTorch
import torch  # PyTorch library


import optuna # Hyperparameter optimization
from optuna.samplers import TPESampler

# HTML and XML parsing
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # HTML and XML parsing
from imblearn.over_sampling import SMOTE  # Handling imbalanced data
from nltk.corpus import stopwords  # Stop words
from nltk.stem import WordNetLemmatizer  # Lemmatization
from nltk.tokenize import word_tokenize  # Tokenization

# Sparse matrices
from scipy.sparse import csr_matrix, hstack  # Sparse matrix operations

# Machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin  # Scikit-learn base classes
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA # Principal Component Analysis

# Ensemble classifiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

# Text feature extraction
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer, CountVectorizer 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score # Model selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.utils import resample  # Resampling utilities
#from xgboost import XGBClassifier

# Spell checking
from spellchecker import SpellChecker  # Spell checking
from torch.utils.data import DataLoader, Dataset  # Data handling in PyTorch

# Progress bar
from tqdm import tqdm  # Progress bar for loops

# Transformers library
# BERT models and training utilities
from transformers import AdamW, BertForSequenceClassification, BertModel, BertTokenizer, Trainer, TrainingArguments
from wordcloud import WordCloud  # Generate word clouds

import pickle  # Pickle (de)serialization
import csv

# Datasets
from datasets import load_dataset  # Load datasets


# Define the mapping of label values to descriptions
label_descriptions = {
    0: "Safe",
    1: "Phishing",
    2: "Spam"
}


# ANSI escape codes for text formatting
BOLD = '\033[1m'
RESET = '\033[0m'


# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)



class DatasetProcessor:
    def __init__(self, df, column_name, dataset_name):
        self.df = df
        self.column_name = column_name
        self.dataset_name = dataset_name



    def drop_unnamed_column(self):
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
            logging.info(f"Dropped 'Unnamed: 0' column from {self.dataset_name}.")

        return self.df



    def check_and_remove_missing_values(self):
        check_missing_values = self.df.isnull().sum()
        total_missing_values = check_missing_values.sum()
        logging.info(f"Total missing values: {total_missing_values}")
        logging.info(f"Removing missing values from {self.dataset_name}...")
        self.df = self.df.dropna()
        logging.info(f"Total number of rows after removing missing values from {self.dataset_name}: {self.df.shape[0]}")

        return self.df



    def remove_duplicates(self):
        logging.info(f"Removing duplicate data....")
        num_duplicates_before = self.df.duplicated(subset=[self.column_name], keep=False).sum()
        self.df = self.df.drop_duplicates(subset=[self.column_name], keep='first')
        num_duplicates_after = self.df.duplicated(subset=[self.column_name], keep=False).sum()
        duplicates_removed = num_duplicates_before - num_duplicates_after
        logging.info(f"Total number of rows identified as duplicates based on '{self.column_name}': {num_duplicates_before}")
        logging.info(f"Number of rows removed due to duplication: {duplicates_removed}")

        return self.df



    def process_dataset(self):
        logging.info(f"Total number of rows in {self.dataset_name} DataFrame: {self.df.shape[0]}")
        self.drop_unnamed_column()
        self.check_and_remove_missing_values()
        self.remove_duplicates()
        logging.info(f"Total number of rows remaining in the {self.dataset_name}: {self.df.shape[0]}\n")
        logging.debug(f"{self.dataset_name} after removing duplicates:\n{self.df.head()}\n")

        return self.df


#Extracting headers from spamassassin CEAS dataset
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

    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        from_match = re.search(r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)

        from_header = from_match.group(1) if from_match else None
        to_header = to_match.group(1) if to_match else None
        mail_to_header = mail_to_match.group(1) if mail_to_match else None

        return {'From': from_header, 'To': to_header, 'Mail-To': mail_to_header}

    def extract_body_content(self, email_message: EmailMessage) -> str:
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
        Function to count occurrences of https vs http in the text.
        """
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
        """
        Function to check if text contains any blacklisted keywords.
        """
        keyword_count = 0
        for keyword in blacklisted_keywords:
            keyword_count += len(re.findall(re.escape(keyword), text, re.IGNORECASE))
        return keyword_count

    def detect_url_shorteners(self, links: List[str]) -> List[str]:
        # Common URL shortener domains
        shortener_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 
    'adf.ly', 'bl.ink', 'lnkd.in', 'shorte.st', 'mcaf.ee', 'q.gs', 'po.st', 
    'bc.vc', 's.coop', 'u.to', 'cutt.ly', 't2mio.com', 'rb.gy', 'clck.ru', 
    'shorturl.at', '1url.com', 'hyperurl.co', 'urlzs.com', 'v.gd', 'x.co']  
        short_urls = [link for link in links if any(domain in link for domain in shortener_domains)]
        return short_urls

    def contains_ip_address(self, text: str) -> bool:
        ip_pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'
        return bool(re.search(ip_pattern, text))
    
    def extract_headers_spamassassin(self) -> pd.DataFrame:
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

                    # Count blacklisted keywords in the email body
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
                    headers_list.append(
                        {'sender': None, 'receiver': None, 'mailto': None, 'texturls': [], 'blacklisted_keywords_count': 0, 'short_urls': [], 'has_ip_address': False})

            self.headers_df = pd.DataFrame(headers_list)
            self.headers_df['texturls'] = self.headers_df['texturls'].apply(self.clean_links)
            self.headers_df['short_urls'] = self.headers_df['short_urls'].apply(self.clean_links)


            return self.headers_df

#CEAS dataset
    def extract_headers_ceas(self) -> pd.DataFrame:
            headers_list: List[Dict[str, int]] = []
            
            for email_text in tqdm(self.df['body'], desc="Extracting headers"):
                try:
                    # Extract body content directly if already in the text field
                    body_content = email_text  # Assuming 'email_text' contains the email body directly
                    logging.debug(f"Email body content: {body_content}")

                    # Count blacklisted keywords and http/https occurrences in the email body
                    https_http_counts = self.count_https_http(body_content)
                    blacklisted_keyword_count = self.contains_blacklisted_keywords(body_content)
                    short_urls = self.detect_url_shorteners(self.clean_links(re.findall(r'https?:\/\/[^\s\'"()<>]+', body_content)))
                    has_ip_address = self.contains_ip_address(body_content)

                    headers_list.append({
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address
                })
                except Exception as e:
                    logging.error(f"Error processing email: {e}")
                    headers_list.append({
                        'https_count': 0,
                        'http_count': 0,
                        'blacklisted_keywords_count': 0,
                        'short_urls': [],
                        'has_ip_address': False
                    })

            self.headers_df = pd.DataFrame(headers_list)
            self.headers_df['short_urls'] = self.headers_df['short_urls'].apply(self.clean_links)
            return self.headers_df


    def save_to_csv(self, file_path: str):
        if not self.headers_df.empty:
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError(
                "No header information extracted. Please run extract_headers() first.")



class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.common_words = set(self.spell_checker.word_frequency.keys())
        self.enable_spell_check = enable_spell_check
        logging.info("Initializing TextProcessor...")



    def expand_contractions(self, text):
        return contractions.fix(text)



    def remove_punctuation(self, text):
        extra_punctuation = '“”‘’—–•·’'
        all_punctuation = string.punctuation + extra_punctuation
        return text.translate(str.maketrans('', '', all_punctuation))



    def tokenize(self, text):
        return word_tokenize(text)



    def remove_stop_words(self, words_list):
        return [w for w in words_list if w.lower() not in self.stop_words]



    def lemmatize(self, words_list):
        return [self.lemmatizer.lemmatize(w) for w in words_list]



    def remove_urls(self, text):
        return re.sub(r'(http[s]?|ftp):\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # Updated to remove non-standard URL formats (e.g., 'http ww newsisfree com')



    def remove_custom_urls(self, text):
        return re.sub(r'\b(?:http|www)[^\s]*\b', '', text)  # Catch patterns like 'http ww' or 'www.' that are incomplete



    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)



    def remove_all_html_elements(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {}

        return soup.get_text(separator=" ", strip=True)



    def remove_email_headers(self, text):
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:', 'Reply-To:', 'Content-Type:', 'Return-Path:', 'Message-ID:',
                   'Received:', 'MIME-Version:', 'Delivered-To:', 'Authentication-Results:', 'DKIM-Signature:', 'X-', 'Mail-To:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)

        return text



    def remove_emails(self, text):
        email_pattern_with_spaces = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' # Regex pattern to match emails with or without spaces around "@"
        email_pattern_no_spaces = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' # Regex pattern to match emails without spaces
        combined_pattern = f"({email_pattern_with_spaces}|{email_pattern_no_spaces})" # Combine both patterns using the OR operator

        return re.sub(combined_pattern, '', text)



    def remove_time(self, text):
        time_pattern = r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?: ?[APMapm]{2})?(?: [A-Z]{1,5})?\b'  # Regex to match various time patterns

        return re.sub(time_pattern, '', text)



    def remove_months(self, text):
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        months_regex = r'\b(?:' + '|'.join(months) + r')\b'

        return re.sub(months_regex, '', text, flags=re.IGNORECASE)



    def remove_dates(self, text):
        date_pattern = (
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|' # Example: Mon, 2 Sep 2002
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|' # Example: 20-09-2002, Sep 13 2002
            r'\b(?:\d{1,2}\s[A-Za-z]+\s\d{4})\b|'  # Example: 01 September 2002
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})\b'  # Example: 24/08/2002
        )

        return re.sub(date_pattern, '', text, flags=re.IGNORECASE)



    def remove_timezones(self, text):
        timezone_pattern = r'\b(?:[A-Z]{2,4}[+-]\d{2,4}|UTC|GMT|PST|EST|CST|MST)\b' # Regex to match time zones (e.g., PST, EST, GMT, UTC)

        return re.sub(timezone_pattern, '', text)



    def remove_multiple_newlines(self, text):
        return re.sub(r'\n{2,}', '\n', text)  # Replace multiple newlines with a single newline



    def remove_words(self, text):
        return re.sub(r'\b(url|original message)\b', '', text, flags=re.IGNORECASE) # Combine both words using the | (OR) operator in regex



    def remove_single_characters(self, text):
        return re.sub(r'\b\w\b', '', text) # Remove single characters that are not part of a word



    def remove_repetitive_patterns(self, text):
        return re.sub(r'\b(nt+ts?|n+|t+|nt+)\b', '', text) # Combine patterns for 'nt+ts?', repetitive 'n' or 'nt', and 't+', 'n+', 'nt+'



    def lowercase_text(self, text):
        return text.lower()



    def remove_bullet_points_and_symbols(self, text):
        symbols = ['•', '◦', '◉', '▪', '▫', '●', '□', '■', '✦', '✧', '✪', '✫', '✬', '✭', '✮', '✯', '✰'] # List of bullet points and similar symbols
        for symbol in symbols:
            text = text.replace(symbol, '')

        return text



    def fit(self, X, y=None):
        return self



    def transform(self, X, y=None):
        cleaned_text_list = []



        for body in tqdm(X, desc='Cleaning Text', unit='email'):
            try:
                text = self.remove_all_html_elements(body)
                text = self.expand_contractions(text)
                text = self.remove_email_headers(text)
                text = self.remove_emails(text)
                text = self.remove_time(text)
                text = self.remove_months(text)
                text = self.remove_dates(text)
                text = self.remove_timezones(text)
                text = self.remove_numbers(text)
                text = self.remove_multiple_newlines(text)
                text = self.remove_custom_urls(text)
                text = self.remove_urls(text)
                text = self.remove_punctuation(text)
                text = self.remove_words(text)
                text = self.remove_single_characters(text)
                text = self.remove_repetitive_patterns(text)
                text = self.lowercase_text(text)
                text = self.remove_bullet_points_and_symbols(text)
                words_list = self.tokenize(text)
                words_list = self.remove_stop_words(words_list)
                lemmatized_list = self.lemmatize(words_list)
                cleaned_text_list.append(' '.join(lemmatized_list))
            except Exception as e:
                logging.error(f"Error processing text: {e}")
                cleaned_text_list.append('')
        if y is not None:
            logging.info(f"Total amount of text processed: {len(cleaned_text_list)}")

            return pd.DataFrame({'cleaned_text': cleaned_text_list, 'label': y})
        else:
            logging.info(f"Total amount of text processed: {len(cleaned_text_list)}")

            return pd.DataFrame({'cleaned_text': cleaned_text_list})



    def save_to_csv_cleaned(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")



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


# Requires update
def visualize_data(df, df_remove_duplicate):
    logging.info("Visualizing data...")
    label_map = {0: 'Safe', 1: 'Phishing', 2: 'Spam'}


    # Original DataFrame counts
    original_label_counts = df['label'].value_counts()
    original_safe_count = original_label_counts.get(0, 0)
    original_phishing_count = original_label_counts.get(1, 0)
    original_spam_count = original_label_counts.get(2, 0)
    original_total_count = original_safe_count + \
        original_phishing_count + original_spam_count


    # Cleaned DataFrame counts
    cleaned_label_counts = df_remove_duplicate['label'].value_counts()
    cleaned_safe_count = cleaned_label_counts.get(0, 0)
    cleaned_phishing_count = cleaned_label_counts.get(1, 0)
    cleaned_spam_count = cleaned_label_counts.get(2, 0)
    cleaned_total_count = cleaned_safe_count + \
        cleaned_phishing_count + cleaned_spam_count
    if original_total_count == 0 or cleaned_total_count == 0:
        logging.warning("No data to visualize.")

        return


    # Filter out labels with 0% for original and cleaned data
    original_data = [(original_safe_count / original_total_count, 'Safe Emails', original_safe_count),
                     (original_phishing_count / original_total_count,
                      'Phishing Emails', original_phishing_count),
                     (original_spam_count / original_total_count, 'Spam Emails', original_spam_count)]
    original_data = [item for item in original_data if item[0] > 0]
    cleaned_data = [(cleaned_safe_count / cleaned_total_count, 'Safe Emails', cleaned_safe_count),
                    (cleaned_phishing_count / cleaned_total_count,
                     'Phishing Emails', cleaned_phishing_count),
                    (cleaned_spam_count / cleaned_total_count, 'Spam Emails', cleaned_spam_count)]
    cleaned_data = [item for item in cleaned_data if item[0] > 0]


    # Plot distribution of safe, phishing, and spam emails in the original and cleaned DataFrames
    fig, axs = plt.subplots(1, 2, figsize=(24, 10))


    # Original DataFrame pie chart
    if original_data:
        original_sizes, original_labels, original_counts = zip(*original_data)
        wedges, texts, autotexts = axs[0].pie(original_sizes, labels=original_labels, autopct='%.0f%%', colors=['blue', 'red', 'green'], startangle=140, textprops={'fontsize': 14, 'color': 'black'})
        axs[0].set_title('Distribution of Safe, Phishing, and Spam Emails (Original)', color='black')
        for i, autotext in enumerate(autotexts):
            autotext.set_text(f'{autotext.get_text()}\n({original_counts[i]})')


    # Cleaned DataFrame pie chart
    if cleaned_data:
        cleaned_sizes, cleaned_labels, cleaned_counts = zip(*cleaned_data)
        wedges, texts, autotexts = axs[1].pie(cleaned_sizes, labels=cleaned_labels, autopct='%.0f%%', colors=['blue', 'red', 'green'], startangle=140, textprops={'fontsize': 14, 'color': 'black'})
        axs[1].set_title('Distribution of Safe, Phishing, and Spam Emails (After Removing Duplicates)', color='black')
        for i, autotext in enumerate(autotexts):
            autotext.set_text(f'{autotext.get_text()}\n({cleaned_counts[i]})')
    plt.show()


    # Plot count of safe, phishing, and spam emails in the original and cleaned DataFrames side by side
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(3)
    bar1 = ax.bar(index, [original_safe_count, original_phishing_count,
                  original_spam_count], bar_width, label='Original', color='blue')
    bar2 = ax.bar(index + bar_width, [cleaned_safe_count, cleaned_phishing_count,
                  cleaned_spam_count], bar_width, label='Removed Duplicate', color='red')
    ax.set_xlabel('Label Type', color='black')
    ax.set_ylabel('Count', color='black')
    ax.set_title('Safe vs Phishing vs Spam Email Count (Original vs Remove Duplicates)', color='black')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Safe', 'Phishing', 'Spam'], color='black')
    ax.legend()
    for p in bar1 + bar2:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black')
    plt.show()



def data_cleaning(dataset_name, df_processed, text_column, clean_file):
    logging.info(f"Text processing {dataset_name} dataset...")
    processor = TextProcessor()
    df_clean = processor.transform(df_processed[text_column], df_processed['label'])
    processor.save_to_csv_cleaned(df_clean, clean_file)
    logging.info("Text processing and saving completed.")
    logging.info(f"DataFrame columns after data cleaning: {df_clean.columns}")

    return df_clean


# Requires update
def plot_word_cloud(text_list, title, width=1500, height=1000, background_color='white', max_words=300, stopwords=None, colormap='viridis', save_to_file=None):
    try:
        logging.info(f"Generating word cloud for {title}...")


        # Ensure text_list is not empty
        if text_list.empty:
            raise ValueError("text_list is empty. Cannot generate word cloud.")


        # Initialize stopwords if None
        if stopwords is None:
            stopwords = set()
        unique_string = " ".join(text_list)
        wordcloud = WordCloud(width=width, height=height, background_color=background_color, max_words=max_words, stopwords=stopwords, colormap=colormap).generate(unique_string)
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=20)
        plt.show()
        if save_to_file:
            wordcloud.to_file(save_to_file)
            logging.info(f"Word cloud saved to {save_to_file}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")



def load_or_clean_data(dataset_name, df, text_column, file_path, cleaning_function):
    logging.info(f"Loading or cleaning data...")
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading from file.")
        df_clean = pd.read_csv(file_path)
        df_clean['cleaned_text'] = df_clean['cleaned_text'].astype(str).fillna('')
        texts = df_clean['cleaned_text'].tolist()
        if not isinstance(texts, (list, tuple)):
            raise ValueError("Input should be a list or tuple of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Input should be a list or tuple of strings. Found non-string elements.")
        
        return df_clean
    else:
        logging.info(f"File {file_path} does not exist. Cleaning data.")
        cleaned_df = cleaning_function(dataset_name, df, text_column, file_path)
        logging.info(f"Data cleaning and saving to {file_path} completed.")

        return cleaned_df
   


def load_or_extract_headers(df: pd.DataFrame, file_path: str, extractor_class, dataset_type: str) -> pd.DataFrame:
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
            logging.info(f"File {file_path} already exists. Loading from file.")
            return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist. Extracting headers for dataset: {dataset_type}.")
        
        header_extractor = extractor_class(df)
        
        # Check dataset type and call the corresponding extraction function
        if dataset_type == "spamassassin":
            headers_df = header_extractor.extract_headers_spamassassin()
            
        elif dataset_type == "ceas":
            headers_df = header_extractor.extract_headers_ceas()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Please specify either 'spamassassin' or 'ceas'.")
        
        header_extractor.save_to_csv(file_path)
        logging.info(f"Email header extraction and saving to {file_path} completed for dataset: {dataset_type}.\n")
        
        return headers_df
    


# Redundant function
def extract_features_for_fold(X_train, X_test, fold_idx, feature_column='cleaned_text', feature_path_prefix="Extracted Body Features"):
    feature_extractor = BERTFeatureExtractor()
    train_texts = X_train[feature_column].tolist()
    test_texts = X_test[feature_column].tolist()
    

    # Define paths for saving features
    if not os.path.exists(feature_path_prefix):
        os.makedirs(feature_path_prefix)
    train_features_path = os.path.join(feature_path_prefix, f"fold_{fold_idx}_train_features.npy")
    test_features_path = os.path.join(feature_path_prefix, f"fold_{fold_idx}_test_features.npy")
    

    # Extract and save features if not already saved
    if not os.path.exists(train_features_path):
        logging.info(f"Extracting BERT features for Fold {fold_idx} - Training data...")
        train_features = feature_extractor.extract_features(train_texts)
        np.save(train_features_path, train_features)
        logging.info(f"Saved train features for Fold {fold_idx} to {train_features_path}.")
    else:
        logging.info(f"Loading saved train features for Fold {fold_idx} from {train_features_path}.")
        train_features = np.load(train_features_path)
    if not os.path.exists(test_features_path):
        logging.info(f"Extracting BERT features for Fold {fold_idx} - Test data...")
        test_features = feature_extractor.extract_features(test_texts)
        np.save(test_features_path, test_features)
        logging.info(f"Saved test features for Fold {fold_idx} to {test_features_path}.")
    else:
        logging.info(f"Loading saved test features for Fold {fold_idx} from {test_features_path}.")
        test_features = np.load(test_features_path)
    return train_features, test_features


# Redundant function
def extract_features_with_column_transformer(X_train, X_test, categorical_columns, numerical_columns, fold_idx, feature_path_prefix="Extracted Other Features"):
    if not os.path.exists(feature_path_prefix):
        os.makedirs(feature_path_prefix)
    train_features_path = os.path.join(feature_path_prefix, f"fold_{fold_idx}_train_encoded.npy")
    test_features_path = os.path.join(feature_path_prefix, f"fold_{fold_idx}_test_encoded.npy")

    # Create the ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
            ('num', StandardScaler(), numerical_columns)  # Apply scaling to numerical columns
        ],
        remainder='passthrough'  # This will keep other columns as they are
    )

    # Check if features are already saved
    if not os.path.exists(train_features_path):
        logging.info(f"Preprocessing features for Fold {fold_idx} - Training data...")
        X_train_encoded = preprocessor.fit_transform(X_train)
        np.save(train_features_path, X_train_encoded)
        logging.info(f"Saved train features for Fold {fold_idx} to {train_features_path}.")
    else:
        logging.info(f"Loading saved train features for Fold {fold_idx} from {train_features_path}.")
        X_train_encoded = np.load(train_features_path, allow_pickle=True)
    if not os.path.exists(test_features_path):
        logging.info(f"Preprocessing features for Fold {fold_idx} - Test data...")
        X_test_encoded = preprocessor.transform(X_test)
        np.save(test_features_path, X_test_encoded)
        logging.info(f"Saved test features for Fold {fold_idx} to {test_features_path}.")
    else:
        logging.info(f"Loading saved test features for Fold {fold_idx} from {test_features_path}.")
        X_test_encoded = np.load(test_features_path, allow_pickle=True)
    return X_train_encoded, X_test_encoded


# Redundant function
def combine_features_and_labels(bert_train_features, other_train_features, bert_test_features, other_test_features, y_train, y_test, fold_idx, combined_features_path_prefix="Combined Features"):
    # Convert features to NumPy arrays
    bert_train_features = np.array(bert_train_features)
    other_train_features = np.array(other_train_features)
    bert_test_features = np.array(bert_test_features)
    other_test_features = np.array(other_test_features)
    
    # Convert labels to NumPy arrays and reshape them
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    
    # Define paths for saving combined features and labels
    if not os.path.exists(combined_features_path_prefix):
        os.makedirs(combined_features_path_prefix)
    combined_train_features_path = os.path.join(combined_features_path_prefix, f"fold_{fold_idx}_combined_train_features.npy")
    combined_test_features_path = os.path.join(combined_features_path_prefix, f"fold_{fold_idx}_combined_test_features.npy")
    
    # Check if combined features and labels are already saved
    if not os.path.exists(combined_train_features_path):
        # Combine features (horizontal stack) if not already saved
        logging.info(f"Combining features and labels for Fold {fold_idx} - Training data...")
        combined_train_features = np.hstack((bert_train_features, other_train_features))
        combined_test_features = np.hstack((bert_test_features, other_test_features))
        combined_train = np.hstack((combined_train_features, y_train))
        combined_test = np.hstack((combined_test_features, y_test))
        
        # Save the combined features and labels
        np.save(combined_train_features_path, combined_train)
        logging.info(f"Saved combined train features and labels for Fold {fold_idx} to {combined_train_features_path}.")
    else:
        logging.info(f"Loading saved combined train features and labels for Fold {fold_idx} from {combined_train_features_path}.")
        combined_train = np.load(combined_train_features_path, allow_pickle=True)
        if combined_train is None:
            raise ValueError(f"Loaded data from {combined_train_features_path} is None.")
    
    if not os.path.exists(combined_test_features_path):
        # Combine features (horizontal stack) if not already saved
        logging.info(f"Combining features and labels for Fold {fold_idx} - Test data...")
        combined_train_features = np.hstack((bert_test_features, other_test_features))
        combined_test = np.hstack((combined_test_features, y_test))
        
        # Save the combined features and labels
        np.save(combined_test_features_path, combined_test)
        logging.info(f"Saved combined test features and labels for Fold {fold_idx} to {combined_test_features_path}.")
    else:
        logging.info(f"Loading saved combined test features and labels for Fold {fold_idx} from {combined_test_features_path}.")
        combined_test = np.load(combined_test_features_path, allow_pickle=True)
        if combined_test is None:
            raise ValueError(f"Loaded data from {combined_test_features_path} is None.")
    
    return combined_train, combined_test



def stratified_k_fold_split(df, n_splits=3, random_state=42):
    logging.info("Performing Stratified K-Fold splitting...")
    

    # Check if DataFrame contains necessary columns
    columns_to_use = ['sender', 'receiver', 'urls', 'cleaned_text', 'label']
    if not set(columns_to_use).issubset(df.columns):
        missing_cols = set(columns_to_use) - set(df.columns)
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    

    # Select columns to use for splitting
    df = df[columns_to_use]
    X = df.drop(columns=['label'])
    y = df['label']
    

    # Perform Stratified K-Fold splitting
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        logging.info(f"Processing Fold {fold_idx}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    logging.info("Completed Stratified K-Fold splitting.")

    return folds


# Redundant function
def pca_incremental(X_train, y_train, n_components=40, batch_size=1000):
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logging.info(f"Shape of scaled data: {X_train_scaled.shape}")
    logging.info("Applying Incremental PCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch in range(0, X_train_scaled.shape[0], batch_size):
        X_batch = X_train_scaled[batch:batch + batch_size]
        ipca.partial_fit(X_batch)
    X_train_pca = ipca.transform(X_train_scaled)
    logging.info(f"Shape of PCA-transformed data: {X_train_pca.shape}")

    return X_train_pca, y_train



def smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced



def load_or_save_model(model, model_path, action='load'):
    if action == 'load':
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
        else:
            logging.info(f"No saved model found at {model_path}. Proceeding to train a new model.")
            return None
    elif action == 'save':
        logging.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)



def load_or_save_params(params, params_path, action='load'):
    if action == 'load':
        if os.path.exists(params_path):
            logging.info(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f)
        else:
            logging.info(f"No saved parameters found at {params_path}. Proceeding to conduct a study.")
            return None
    elif action == 'save':
        logging.info(f"Saving parameters to {params_path}")
        with open(params_path, 'w') as f:
            json.dump(params, f)



def model_training(X_train, y_train, X_test, y_test, model_path, params_path):
    # Check if ensemble model and parameters exist
    ensemble_model = load_or_save_model(None, model_path, action='load')
    best_params = load_or_save_params(None, params_path, action='load')


    # Train a new model if no existing model or parameters are found
    if ensemble_model is None and best_params is None:
        logging.info("No existing ensemble model or parameters found. Conducting Optuna study and training model...")
        best_params = conduct_optuna_study(X_train, y_train)
        load_or_save_params(best_params, params_path, action='save')
        ensemble_model = train_ensemble_model(best_params, X_train, y_train, model_path)
    elif ensemble_model is None and best_params is not None:
        logging.info("Parameters found, but no ensemble model. Training new model with existing parameters...")
        ensemble_model = train_ensemble_model(best_params, X_train, y_train, model_path)
    elif ensemble_model is not None and best_params is None:
        logging.info("Ensemble model found, but no parameters. Using pre-trained model for evaluation.")
    else:
        logging.info("Ensemble model and parameters found. Using pre-trained model for evaluation.")


    # Make predictions
    y_train_pred = ensemble_model.predict(X_train)
    y_test_pred = ensemble_model.predict(X_test)


    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Safe', 'Phishing', 'Spam']


    # Print the performance metrics
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_test_pred)))
    print("Classification Report for Training Data:\n" + classification_report(y_train, y_train_pred, target_names=target_names))
    print("\nClassification Report for Test Data:" + classification_report(y_test, y_test_pred, target_names=target_names))
    
    return ensemble_model, test_accuracy


def conduct_optuna_study(X_train, y_train):
    # Define the objective function for Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 10, 100, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)


        # Define the RandomForestClassifier with suggested hyperparameters
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42,
            n_jobs=2
        )


        # Perform cross-validation to get the average accuracy score
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        return accuracy_score(y_train, y_train_pred)


    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=5)
    logging.info(f"Best hyperparameters: {study.best_params}")

    return study.best_params



def train_ensemble_model(best_params, X_train, y_train, model_path):
    logging.info(f"Training new ensemble model with best parameters: {best_params}")


    # Define the base models for the ensemble
    logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    gbm_model = GradientBoostingClassifier(random_state=42)


    # Train RandomForest with the best hyperparameters
    best_rf_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        class_weight='balanced',
        random_state=42,
        n_jobs=2
    )
    best_rf_model.fit(X_train, y_train)


    # Create the ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ('rf', best_rf_model),
        ('logreg', logreg_model)
        #('gbm', gbm_model)
    ], voting='soft')


    # Train the ensemble model
    for _ in tqdm(range(1), desc="Training ensemble model"):
        ensemble_model.fit(X_train, y_train)


    # Save the ensemble model
    load_or_save_model(ensemble_model, model_path, action='save')
    logging.info(f"Ensemble model trained and saved to {model_path}.\n")
    
    return ensemble_model



def log_label_percentages(df, dataset_name):
    total_count = len(df)
    total_rows, total_columns = df.shape
    label_counts = df['label'].value_counts(normalize=True) * 100
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Total count: {total_count}")
    logging.info(f"Total rows: {total_rows}")
    logging.info(f"Total columns: {total_columns}")
    sorted_label_counts = label_counts.sort_index()
    num_labels = len(sorted_label_counts)
    for i, (label, percentage) in enumerate(sorted_label_counts.items()):
        description = label_descriptions.get(label, "Unknown")
        if i == num_labels - 1:
            logging.info(f"{description} percentage: {percentage:.2f}%\n")
        else:
            logging.info(f"{description} percentage: {percentage:.2f}%")



def count_urls(urls_list):
    if isinstance(urls_list, list):
        return len(urls_list)
    else:
        return 0



def check_missing_values(df, df_name, num_rows=1):
    missing_values = df.isnull().sum()
    total_missing_values = missing_values.sum()
    if total_missing_values == 0:
        logging.info(f"No missing values in {df_name}.")
    else:
        logging.info(f"Total missing values in {df_name}: {total_missing_values}")
        columns_with_missing = missing_values[missing_values > 0]
        logging.info(f"Columns with missing values in {df_name}:")
        for column, count in columns_with_missing.items():
            logging.info(f"Column '{column}': {count} missing values")
        rows_with_missing = df[df.isnull().any(axis=1)]
        if rows_with_missing.empty:
            logging.info(f"No rows with missing values found in {df_name} after initial check.")



def get_fold_paths(fold_idx, base_dir='Processed Data'):
    train_data_path = os.path.join(base_dir, f"fold_{fold_idx}_train_data.npz")
    test_data_path = os.path.join(base_dir, f"fold_{fold_idx}_test_data.npz")
    train_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_train_labels.pkl")
    test_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_test_labels.pkl")
    
    return train_data_path, test_data_path, train_labels_path, test_labels_path



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
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Processed Data')
    os.makedirs(base_dir, exist_ok=True)
    train_data_path, test_data_path, train_labels_path, test_labels_path = get_fold_paths(fold_idx, base_dir)


    # Check if the files already exist
    if not all([os.path.exists(train_data_path), os.path.exists(test_data_path), os.path.exists(train_labels_path), os.path.exists(test_labels_path)]):
        logging.info(f"Running pipeline for fold {fold_idx}...")


        # Fit and transform the pipeline
        X_train_non_text_processed = pipeline.named_steps['preprocessor'].fit_transform(X_train.drop(columns=['cleaned_text']))
        X_test_non_text_processed = pipeline.named_steps['preprocessor'].transform(X_test.drop(columns=['cleaned_text']))


        # Transform the text features
        X_train_text_processed = pipeline.named_steps['bert_features'].transform(X_train['cleaned_text'].tolist())
        X_test_text_processed = pipeline.named_steps['bert_features'].transform(X_test['cleaned_text'].tolist())


        # Combine processed features
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
        # Load the preprocessed data
        logging.info(f"Loading preprocessed data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = load_data_pipeline(train_data_path, train_labels_path)
        X_test_combined, y_test = load_data_pipeline(test_data_path, test_labels_path)

    return X_train_balanced, X_test_combined, y_train_balanced, y_test



# Main processing function
def main():
    # Use relative paths to access the datasets and save the extracted data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'CEAS_08.csv')
    ExtractedSpamAssassinEmailHeaderFile = os.path.join(base_dir, 'Extracted Data', 'SpamAssassinExtractedEmailHeader.csv')
    ExtractedCEASEmailHeaderFile = os.path.join(base_dir, 'Extracted Data', 'CEASExtractedEmailHeader.csv')
    MergedSpamAssassinFile = os.path.join(base_dir, 'Extracted Data', 'MergedSpamAssassin.csv')
    MergedDataFrame = os.path.join(base_dir, 'Extracted Data', 'MergedDataFrame.csv')
    CleanedDataFrame = os.path.join(base_dir, 'Extracted Data', 'CleanedDataFrame.csv')
    

    SavedModel = os.path.join(base_dir, 'Model & Parameters', 'EnsembleModel.pkl')
    SavedParameters = os.path.join(base_dir, 'Model & Parameters', 'BestParameters.json')


    #BertFeatures = os.path.join(base_dir, 'Extracted Features')
    #CleanedCeasFile = os.path.join(base_dir, 'Extracted Data', 'CleanedCEAS_08Text.csv')
    #CleanedSpamAssassinFile = os.path.join(base_dir, 'Extracted Data', 'CleanedSpamAssassinText.csv')
    #MergedHeaderDataset = os.path.join(base_dir, 'Extracted Data', 'MergedHeaderDataset.csv')
    #CEASExtractedFeatures = os.path.join(base_dir, 'Extracted Features', 'CEASExtractedBertFeatures.npy')
    #SpamAssassinExtractedFeatures = os.path.join(base_dir, 'Extracted Features', 'SpamAssassinExtractedBertFeatures.npy')

    # Load the datasets
    df_ceas = pd.read_csv(dataset)
    dataset = load_dataset('talby/spamassassin',split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()


    try:
        '''
        Data Preprocessing
        '''
        logging.info(f"Beginning data preprocessing...")


        # Change label values to match the labeling scheme
        df_spamassassin['label'] = df_spamassassin['label'].map({1: 0, 0: 2})


        # Remove duplicates and missing values
        processor_spamassassin = DatasetProcessor(df_spamassassin, 'text', 'SpamAssassin')
        df_processed_spamassassin = processor_spamassassin.process_dataset()
        processor_ceas = DatasetProcessor(df_ceas, 'body', 'CEAS_08')
        df_processed_ceas = processor_ceas.process_dataset()


        # Combined DataFrame
        combined_percentage_df = pd.concat([df_processed_spamassassin, df_processed_ceas])


        # Check if DataFrame has merged correctly
        log_label_percentages(df_processed_ceas, 'CEAS_08')
        log_label_percentages(df_processed_spamassassin, 'SpamAssassin')
        log_label_percentages(combined_percentage_df,'Combined CEAS_08 and SpamAssassin')
        check_missing_values(combined_percentage_df, 'Combined CEAS_08 and SpamAssassin')
        logging.info(f"Data Preprocessing completed.\n")
        #Columns in CEAS_08 dataset: ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
        #Columns in SpamAssassin dataset: ['label', 'group', 'text']
        

        '''
        Feature Engineering
        '''
        logging.info(f"Beginning feature engineering...")


        # Extract email headers from the SpamAssassin dataset
        spamassassin_headers_df = load_or_extract_headers(df_processed_spamassassin, ExtractedSpamAssassinEmailHeaderFile, EmailHeaderExtractor, 'spamassassin')
        print(spamassassin_headers_df.columns.to_list())
        logging.info("Email header extraction and saving from Spam Assassin completed.")
        # Columns in current extracted email headers: ['sender', 'receiver', 'mailto', 'texturls']
        spamassassin_headers_df['urls'] = spamassassin_headers_df['texturls'].apply(count_urls) # Convert text to number for URLs
        # Columns in current extracted email headers: ['sender', 'receiver', 'mailto', 'urls']
        spamassassin_headers_df.drop(columns=['mailto'], inplace=True) # Drop the 'mailto' column
        spamassassin_headers_df.drop(columns=['texturls'], inplace=True) # Drop the 'texturls' column
        # Columns in current extracted email headers: ['sender', 'receiver', 'urls']
        ceas_headers_df = load_or_extract_headers(df_processed_ceas, ExtractedCEASEmailHeaderFile, EmailHeaderExtractor, 'ceas')
        print(ceas_headers_df.columns.to_list())
        #logging.info("Email header extraction and saving from CEAS completed.")
        logging.info(f"Feature engineering completed.\n")


        '''
        Data Integration
        '''
        logging.info(f"Beginning data integration...")


        # Merging the processed SpamAssassin dataset with the extracted email headers
        df_processed_spamassassin.reset_index(inplace=True)
        spamassassin_headers_df.reset_index(inplace=True)
        spamassassin_headers_df.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
        if len(df_processed_spamassassin) == len(spamassassin_headers_df):
            merged_spamassassin_df = pd.merge(df_processed_spamassassin, spamassassin_headers_df, on='index', how='left')
            merged_spamassassin_df = merged_spamassassin_df.rename(columns={'text': 'body'})
            merged_spamassassin_df = merged_spamassassin_df[['sender', 'receiver', 'urls', 'body', 'label', 'index']]
            missing_in_merged_df = merged_spamassassin_df[merged_spamassassin_df['index'].isnull()]
            logging.info(f"Number of missing rows in Merged Spam Assassin Dataframe: {len(missing_in_merged_df)}")
            logging.info(f'Total rows in Processed Spam Assassin Dataframe: {len(df_processed_spamassassin)}')
            logging.info(f"Total rows in Merged Spam Assassin Dataframe: {len(merged_spamassassin_df)}")
            merged_spamassassin_df.drop(columns=['index'], inplace=True)
            # Columns in merged_spamassassin_df: ['sender', 'receiver', 'urls', 'body', 'label']
        else:
            logging.error("The number of rows in Processed Spam Assassin and Spam Assassin Header do not match.")
            raise ValueError("The number of rows in Processed Spam Assassin and Spam Assassin Header do not match.")


        # Verifying the merged SpamAssassin DataFrame
        if len(merged_spamassassin_df) != len(df_processed_spamassassin):
            logging.error("The number of rows in the Merged Spam Assassin DataFrame DataFrame does not match Processed Spam Assassin.")
            raise ValueError("The number of rows in the Merged Spam Assassin DataFrame does not match Processed Spam Assassin.")
        else:
            logging.info("The number of rows in the Merged Spam Assassin DataFrame matches Processed Spam Assassin.")
            merged_spamassassin_df.to_csv(MergedSpamAssassinFile, index=False)
            logging.info(f"Merged Spam Assassin DataFrame successfully saved to {MergedSpamAssassinFile}")


        # Merge Spam Assassin and CEAS_08 datasets
        common_columns = ['sender', 'receiver', 'urls', 'body', 'label']
        df_spamassassin_common = merged_spamassassin_df[common_columns]
        df_ceas_common = df_processed_ceas[common_columns]
        combined_df = pd.concat([df_spamassassin_common, df_ceas_common])


        # Verifying the combined DataFrame
        combined_labels = set(combined_df['label'].unique())
        percentage_labels = set(combined_percentage_df['label'].unique())
        if combined_labels != percentage_labels:
            logging.error(f"Labels in Merged DataFrame do not match those in Combined Processed DataFrame. "
                        f"Merged DataFrame labels: {combined_labels}, "
                        f"Combined Processed DataFrame labels: {percentage_labels}")
            raise ValueError("Labels do not match between Merged DataFrame and Combined Processed DataFrame.")
        else:
            logging.info("Labels in Merged DataFrame match those in Combined Processed DataFrame.")
        combined_label_counts = combined_df['label'].value_counts().sort_index()
        percentage_label_counts = combined_percentage_df['label'].value_counts().sort_index()
        if not combined_label_counts.equals(percentage_label_counts):
            logging.error("Label distributions in Merged DataFrame do not match those in Combined Processed DataFrame.")
            logging.error(f"Merged DataFrame distributions:\n{combined_label_counts}")
            logging.error(f"Combined Processed DataFrame distributions:\n{percentage_label_counts}")
            raise ValueError("Label distributions do not match between Merged DataFrame and Combined Processed DataFrame.")
        else:
            logging.info("Label distributions in Merged DataFrame match those in Combined Processed DataFrame.")


        # Save the Merged DataFrame
        combined_df.to_csv(MergedDataFrame, index=False)
        # Columns in Merged DataFrame: ['sender', 'receiver', 'urls', 'body', 'label']
        logging.info(f"Merged DataFrame successfully saved to {MergedDataFrame}")
        logging.info(f"Data Integration completed.\n")


        '''
        Data Cleaning
        '''
        logging.info(f"Beginning data cleaning...")
        df_clean = load_or_clean_data('Merged Dataframe', combined_df, 'body', CleanedDataFrame, data_cleaning)


        # Concatenate the Cleaned DataFrame with the Merged DataFrame
        combined_df_reset = combined_df.reset_index(drop=True)
        df_clean_reset = df_clean.reset_index(drop=True)
        df_cleaned_combined = pd.concat([
            combined_df_reset[['sender', 'receiver', 'urls', 'label']],  # Select necessary columns from merged
            df_clean_reset[['cleaned_text']]  # Select the cleaned_text and label from df_clean
        ], axis=1)


        # Verifying the Cleaned Combine DataFrame
        combined_labels = combined_df['label'].unique()
        df_cleaned_combined_labels = df_cleaned_combined['label'].unique()
        if set(combined_labels) != set(df_cleaned_combined_labels):
            logging.error(f"Labels in Combined DataFrame do not match those in Cleaned Combined DataFrame. "
                        f"Combined DataFrame labels: {combined_labels}, "
                        f"Cleaned Combined DataFrame labels: {df_cleaned_combined_labels}")
            raise ValueError("Labels do not match between Combined DataFrame and Cleaned Combined DataFrame.")
        else:
            logging.info("Labels in Combined DataFrame match those in Cleaned Combined DataFrame.")
        combined_label_counts = combined_df['label'].value_counts().sort_index()
        df_cleaned_combined_label_counts = df_cleaned_combined['label'].value_counts().sort_index()
        if not combined_label_counts.equals(df_cleaned_combined_label_counts):
            logging.error("Label distributions in Combined DataFrame do not match those in Cleaned Combined DataFrame.")
            logging.error(f"Combined DataFrame distributions:\n{combined_label_counts}")
            logging.error(f"Cleaned Combined DataFrame distributions:\n{df_cleaned_combined_label_counts}")
            raise ValueError("Label distributions do not match between Combined DataFrame and Cleaned Combined DataFrame.")
        else:
            logging.info("Label distributions in Combined DataFrame match those in Cleaned Combined DataFrame.")


        # Final columns to keep
        df_cleaned_combined = df_cleaned_combined[['sender', 'receiver', 'urls', 'cleaned_text', 'label']]
        logging.info(f"Final combined DataFrame has {len(df_cleaned_combined)} rows and columns: {df_cleaned_combined.columns.tolist()}")
        check_missing_values(df_cleaned_combined, 'Cleaned Combined DataFrame')
        logging.info(f"Data Cleaning completed.\n")

    
        '''
        Data Splitting
        '''
        logging.info(f"Beginning data splitting...")
        folds = stratified_k_fold_split(df_cleaned_combined)
        logging.info(f"Data Splitting completed.\n")


        # Initialize lists to store accuracies for each fold
        fold_train_accuracies = []
        fold_test_accuracies = []



        for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds, start=1):
            '''
            Feature Extraction and Data Imbalance Handling
            '''
            logging.info(f"Beginning feature extraction for Fold {fold_idx}...")


            # Define columns for categorical, numerical, and text data
            categorical_columns = ['sender', 'receiver']
            numerical_columns = ['urls']
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
                remainder='passthrough'  # Keep other columns unchanged
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


            '''
            Model Training and Evaluation
            '''
            # Train the model and evaluate the performance for each fold
            model_path = os.path.join('Models & Parameters', f'ensemble_model_fold_{fold_idx}.pkl')
            params_path = os.path.join('Models & Parameters', f'best_params_fold_{fold_idx}.json')
            ensemble_model, test_accuracy = model_training(
                X_train_balanced,
                y_train_balanced,
                X_test_combined,
                y_test,
                model_path=model_path,
                params_path=params_path
            )
            fold_test_accuracies.append(test_accuracy)
            logging.info(f"Data for Fold {fold_idx} has been processed, model trained, and evaluated.\n")
        
        
        # Calculate and log the overall test accuracy
        mean_test_accuracy = np.mean(fold_test_accuracies)
        logging.info(f"Overall Test Accuracy: {mean_test_accuracy * 100:.2f}%")
                


    except Exception as e:
        logging.error(f"An error occurred: {e}")



# Call the main function
if __name__ == "__main__":
    main()
