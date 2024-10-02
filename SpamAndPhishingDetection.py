# Standard Libraries
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
from typing import Dict, List, Union  # Type hints

# Datasets
from datasets import load_dataset  # Load datasets



# Define the mapping of label values to descriptions
label_descriptions = {
    0: "Safe",
    1: "Not Safe"
}


# ANSI escape codes for text formatting
BOLD = '\033[1m'
RESET = '\033[0m'


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



class DatasetProcessor:
    """
    A class to process datasets by removing unnamed columns, missing values, and duplicates, and saving the processed data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be processed.
    column_name : str
        The column name to check for duplicates.
    dataset_name : str
        The name of the dataset.
    save_path : str
        The path to save the processed data.
    """
    def __init__(self, df, column_name, dataset_name, save_path):
        self.df = df
        self.column_name = column_name
        self.dataset_name = dataset_name
        self.save_path = save_path


    def drop_unnamed_column(self):
        """
        Drop the 'Unnamed: 0' column if it exists in the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the 'Unnamed: 0' column removed if it existed.
        """
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
            logging.info(f"Dropped 'Unnamed: 0' column from {self.dataset_name}.")

        return self.df



    def check_and_remove_missing_values(self):
        """
        Check and remove missing values from the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing values removed.
        """
        check_missing_values = self.df.isnull().sum()
        total_missing_values = check_missing_values.sum()
        logging.info(f"Total missing values: {total_missing_values}")
        logging.info(f"Removing missing values from {self.dataset_name}...")
        self.df = self.df.dropna()
        logging.info(f"Total number of rows after removing missing values from {self.dataset_name}: {self.df.shape[0]}")

        return self.df



    def remove_duplicates(self):
        """
        Remove duplicate rows based on the specified column.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with duplicates removed.
        """
        logging.info(f"Removing duplicate data....")
        num_duplicates_before = self.df.duplicated(subset=[self.column_name], keep=False).sum()
        self.df = self.df.drop_duplicates(subset=[self.column_name], keep='first')
        num_duplicates_after = self.df.duplicated(subset=[self.column_name], keep=False).sum()
        duplicates_removed = num_duplicates_before - num_duplicates_after
        logging.info(f"Total number of rows identified as duplicates based on '{self.column_name}': {num_duplicates_before}")
        logging.info(f"Number of rows removed due to duplication: {duplicates_removed}")

        return self.df



    def save_processed_data(self):
        """
        Save the processed DataFrame to a CSV file.

        Returns
        -------
        None
        """
        try:
            self.df.to_csv(self.save_path, index=False)
            logging.info(f"Processed data saved to {self.save_path}\n")
        except PermissionError as e:
            logging.error(f"Permission denied: {e}")
        except Exception as e:
            logging.error(f"An error occurred while saving the file: {e}")



    def process_dataset(self):
        """
        Process the dataset by dropping unnamed columns, removing missing values, and removing duplicates.

        Returns
        -------
        pandas.DataFrame
            The processed DataFrame.
        """
        if os.path.exists(self.save_path):
            logging.info(f"Processed file already exists at {self.save_path}. Loading the file...")
            self.df = pd.read_csv(self.save_path)
        else:
            logging.info(f"Total number of rows in {self.dataset_name} DataFrame: {self.df.shape[0]}")
            self.drop_unnamed_column()
            self.check_and_remove_missing_values()
            self.remove_duplicates()
            logging.info(f"Total number of rows remaining in the {self.dataset_name}: {self.df.shape[0]}")
            logging.debug(f"{self.dataset_name} after removing duplicates:\n{self.df.head()}\n")
            self.save_processed_data()

        return self.df



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
    


    def extract_headers_spamassassin(self) -> pd.DataFrame:
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



    def extract_headers_ceas(self) -> pd.DataFrame:
        """
        Extract headers and other relevant information from the email data for CEAS dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted headers and information.
        """
        headers_list: List[Dict[str, int]] = []
        
        for email_text in tqdm(self.df['body'], desc="Extracting headers"):
            try:
                body_content = email_text  # Assuming 'email_text' contains the email body directly
                logging.debug(f"Email body content: {body_content}")


                # Count blacklisted keywords and http/https occurrences in the email body
                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.contains_blacklisted_keywords(body_content)
                short_urls = self.detect_url_shorteners(self.clean_links(re.findall(r'https?:\/\/[^\s\'"()<>]+', body_content)))
                has_ip_address = self.count_ip_addresses(body_content)



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
                    'has_ip_address': 0
                })
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



class TextProcessor:
    """
    A class for processing text data with various cleaning and preprocessing steps.

    Parameters
    ----------
    enable_spell_check : bool, optional
        Whether to enable spell checking. Default is False.

    Attributes
    ----------
    stop_words : set
        A set of stop words to be removed from the text.
    lemmatizer : WordNetLemmatizer
        An instance of WordNetLemmatizer for lemmatizing words.
    spell_checker : SpellChecker
        An instance of SpellChecker for spell checking.
    common_words : set
        A set of common words from the spell checker's word frequency.
    enable_spell_check : bool
        Whether spell checking is enabled.
    """
    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.common_words = set(self.spell_checker.word_frequency.keys())
        self.enable_spell_check = enable_spell_check
        logging.info("Initializing TextProcessor...")

    def expand_contractions(self, text):
        """
        Expand contractions in the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text with contractions expanded.
        """
        return contractions.fix(text)

    def remove_punctuation(self, text):
        """
        Remove punctuation from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without punctuation.
        """
        extra_punctuation = '“”‘’—–•·’'
        all_punctuation = string.punctuation + extra_punctuation
        return text.translate(str.maketrans('', '', all_punctuation))

    def tokenize(self, text):
        """
        Tokenize the text into words.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        list
            A list of words.
        """
        return word_tokenize(text)

    def remove_stop_words(self, words_list):
        """
        Remove stop words from the list of words.

        Parameters
        ----------
        words_list : list
            The list of words.

        Returns
        -------
        list
            The list of words without stop words.
        """
        return [w for w in words_list if w.lower() not in self.stop_words]

    def lemmatize(self, words_list):
        """
        Lemmatize the list of words.

        Parameters
        ----------
        words_list : list
            The list of words.

        Returns
        -------
        list
            The list of lemmatized words.
        """
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    def remove_urls(self, text):
        """
        Remove URLs from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without URLs.
        """
        return re.sub(r'(http[s]?|ftp):\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    def remove_custom_urls(self, text):
        """
        Remove custom URL patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without custom URL patterns.
        """
        return re.sub(r'\b(?:http|www)[^\s]*\b', '', text)

    def remove_numbers(self, text):
        """
        Remove numbers from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without numbers.
        """
        return re.sub(r'\d+', '', text)

    def remove_all_html_elements(self, text):
        """
        Remove all HTML elements from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without HTML elements.
        """
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {}
        return soup.get_text(separator=" ", strip=True)

    def remove_email_headers(self, text):
        """
        Remove email headers from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without email headers.
        """
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:', 'Reply-To:', 'Content-Type:', 'Return-Path:', 'Message-ID:',
                   'Received:', 'MIME-Version:', 'Delivered-To:', 'Authentication-Results:', 'DKIM-Signature:', 'X-', 'Mail-To:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)
        return text

    def remove_emails(self, text):
        """
        Remove email addresses from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without email addresses.
        """
        email_pattern_with_spaces = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_pattern_no_spaces = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        combined_pattern = f"({email_pattern_with_spaces}|{email_pattern_no_spaces})"
        return re.sub(combined_pattern, '', text)

    def remove_time(self, text):
        """
        Remove time patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without time patterns.
        """
        time_pattern = r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?: ?[APMapm]{2})?(?: [A-Z]{1,5})?\b'
        return re.sub(time_pattern, '', text)

    def remove_months(self, text):
        """
        Remove month names from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without month names.
        """
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        months_regex = r'\b(?:' + '|'.join(months) + r')\b'
        return re.sub(months_regex, '', text, flags=re.IGNORECASE)

    def remove_dates(self, text):
        """
        Remove date patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without date patterns.
        """
        date_pattern = (
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|'
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|'
            r'\b(?:\d{1,2}\s[A-Za-z]+\s\d{4})\b|'
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})\b'
        )
        return re.sub(date_pattern, '', text, flags=re.IGNORECASE)

    def remove_timezones(self, text):
        """
        Remove time zone patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without time zone patterns.
        """
        timezone_pattern = r'\b(?:[A-Z]{2,4}[+-]\d{2,4}|UTC|GMT|PST|EST|CST|MST)\b'
        return re.sub(timezone_pattern, '', text)

    def remove_multiple_newlines(self, text):
        """
        Remove multiple newlines from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text with multiple newlines replaced by a single newline.
        """
        return re.sub(r'\n{2,}', '\n', text)

    def remove_words(self, text):
        """
        Remove specific words from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without specific words.
        """
        return re.sub(r'\b(url|original message|submissionid|submission)\b', '', text, flags=re.IGNORECASE)

    def remove_single_characters(self, text):
        """
        Remove single characters from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without single characters.
        """
        return re.sub(r'\b\w\b', '', text)

    def remove_repetitive_patterns(self, text):
        """
        Remove repetitive patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without repetitive patterns.
        """
        return re.sub(r'\b(nt+ts?|n+|t+|nt+)\b', '', text)

    def lowercase_text(self, text):
        """
        Convert the text to lowercase.

        Parameters:
        text (str): The input text.

        Returns:
        str: The text in lowercase.
        """
        return text.lower()

    def remove_bullet_points_and_symbols(self, text):
        """
        Remove bullet points and similar symbols from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without bullet points and symbols.
        """
        symbols = ['•', '◦', '◉', '▪', '▫', '●', '□', '■', '✦', '✧', '✪', '✫', '✬', '✭', '✮', '✯', '✰']
        for symbol in symbols:
            text = text.replace(symbol, '')
        return text

    def clean_text(self, X, y=None):
        """
        Clean and preprocess a list of text data.

        Parameters
        ----------
        X : list
            A list of text data to be cleaned.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the cleaned text data.
        """
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
        return pd.DataFrame({'cleaned_text': cleaned_text_list})

    def save_to_csv_cleaned(self, df, filename):
        """
        Save the cleaned text data to a CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the cleaned text data.
        filename : str
            The file path to save the CSV file.

        Returns
        -------
        None
        """
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")



class BERTFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract BERT features from text data.

    Parameters
    ----------
    feature_extractor : BERTFeatureExtractor
        An instance of BERTFeatureExtractor to extract features.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data (no-op).
    transform(X)
        Transforms the data by extracting BERT features.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, X, y=None):
        """
        Fit the transformer to the data (no-op).

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like, optional
            The target values (default is None).

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the data by extracting BERT features.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        numpy.ndarray
            The extracted BERT features.
        """
        return np.array(self.feature_extractor.extract_features(X))
    


class TextDataset(Dataset):
    """
    A custom dataset for handling text data for BERT.

    Parameters
    ----------
    texts : list
        A list of text samples.
    labels : list
        A list of labels corresponding to the text samples.
    tokenizer : BertTokenizer
        An instance of BertTokenizer for tokenizing the text.
    max_length : int
        The maximum length of the tokenized sequences.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Returns a dictionary containing the input IDs, attention mask, and label for the sample at index idx.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length



    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.texts)



    def __getitem__(self, idx):
        """
        Returns a dictionary containing the input IDs, attention mask, and label for the sample at index idx.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the input IDs, attention mask, and label for the sample.
        """
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
    """
    A class to extract BERT features from text data.

    Parameters
    ----------
    max_length : int, optional
        The maximum length of the tokenized sequences. Default is 128.
    device : torch.device, optional
        The device to run the BERT model on. Default is CUDA if available, otherwise CPU.

    Methods
    -------
    extract_features(texts, batch_size=16)
        Extracts BERT features from a list of text samples.
    """
    def __init__(self, max_length=128, device=None):
        logging.info("Initializing BERT Feature Extractor...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Ensure model is on the right device



    def extract_features(self, texts, batch_size=16):
        """
        Extract BERT features from a list of text samples.

        Parameters
        ----------
        texts : list
            A list of text samples.
        batch_size : int, optional
            The batch size for processing the text samples. Default is 16.

        Returns
        -------
        list
            A list of extracted BERT features.
        """
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
    


class RareCategoryRemover(BaseEstimator, TransformerMixin):
    """
    A custom transformer to remove rare categories from categorical features.

    Parameters
    ----------
    threshold : float, optional
        The frequency threshold below which categories are considered rare. Default is 0.05.

    Attributes
    ----------
    replacements_ : dict
        A dictionary mapping each column to another dictionary of rare categories and their replacements.
    """
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.replacements_ = {}



    def fit(self, X, y=None):
        """
        Fit the transformer to the data by identifying rare categories.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data with categorical features.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logging.info(f"Removing rare categories with threshold: {self.threshold}")
        for column in X.columns:
            frequency = X[column].value_counts(normalize=True)
            rare_categories = frequency[frequency < self.threshold].index
            self.replacements_[column] = {cat: 'Other' for cat in rare_categories}

        return self



    def transform(self, X):
        """
        Transform the data by replacing rare categories with 'Other'.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data with categorical features.

        Returns
        -------
        pandas.DataFrame
            The transformed data with rare categories replaced.
        """
        for column, replacements in self.replacements_.items():
            X.loc[:, column] = X[column].replace(replacements)
        assert X.shape[0] == X.shape[0], "Row count changed during rare category removal."

        return X



def data_cleaning(dataset_name, df_processed, text_column, clean_file):
    """
    Cleans the text data in the specified column of the DataFrame.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being processed.
    df_processed : pandas.DataFrame
        The DataFrame containing the processed data.
    text_column : str
        The name of the column containing text data to be cleaned.
    clean_file : str
        The file path where the cleaned data will be saved.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    logging.info(f"Text processing {dataset_name} dataset...")
    processor = TextProcessor()
    df_clean = processor.clean_text(df_processed[text_column], df_processed['label'])
    processor.save_to_csv_cleaned(df_clean, clean_file)
    logging.info("Text processing and saving completed.")
    #logging.info(f"DataFrame columns after data cleaning: {df_clean.columns}")

    return df_clean



def load_or_clean_data(dataset_name, df, text_column, file_path, cleaning_function):
    """
    Loads the data from the specified file path or cleans the data if the file does not exist.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being processed.
    df : pandas.DataFrame
        The DataFrame containing the data.
    text_column : str
        The name of the column containing text data to be cleaned.
    file_path : str
        The file path where the cleaned data will be saved.
    cleaning_function : function
        The function to clean the data.

    Returns
    -------
    pandas.DataFrame
        The loaded or cleaned DataFrame.
    """
    #logging.info(f"Loading or cleaning data...")
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
        #logging.info(f"Data cleaning and saving to {file_path} completed.")

        return cleaned_df
   


def load_or_extract_headers(df: pd.DataFrame, file_path: str, extractor_class, dataset_type: str) -> pd.DataFrame:
    """
    Loads the email headers from the specified file path or extracts them if the file does not exist.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    file_path : str
        The file path where the extracted headers will be saved.
    extractor_class : class
        The class used to extract the headers.
    dataset_type : str
        The type of dataset being processed.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with extracted headers.
    """
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
            logging.info(f"File {file_path} already exists. Loading from file.")

            return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist. Extracting headers for dataset: {dataset_type}.")
        header_extractor = extractor_class(df)
        

        # Check dataset type and call the corresponding extraction function
        if dataset_type == "Spam Assassin":
            headers_df = header_extractor.extract_headers_spamassassin()
        elif dataset_type == "CEAS_08":
            headers_df = header_extractor.extract_headers_ceas()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Please specify either 'Spam Assassin' or 'CEAS_08'.")
        header_extractor.save_to_csv(file_path)
        logging.info(f"Email header extraction and saving to {file_path} completed for dataset: {dataset_type}.")
        
        return headers_df
    


def stratified_k_fold_split(df, n_splits=3, random_state=42, output_dir='Data Splitting'):
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
        X_train_file = os.path.join(output_dir, f'X_train_fold{fold_idx}.csv')
        y_train_file = os.path.join(output_dir, f'y_train_fold{fold_idx}.csv')
        X_test.to_csv(X_test_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        X_train.to_csv(X_train_file, index=False)
        y_train.to_csv(y_train_file, index=False)
        folds.append((X_train, X_test, y_train, y_test))
    logging.info("Completed Stratified K-Fold splitting.")

    return folds



def smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.
    random_state : int, optional
        The random state for reproducibility. Default is 42.

    Returns
    -------
    tuple
        The balanced training data features and labels.
    """
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced



def load_or_save_model(model, model_path, action='load'):
    """
    Loads or saves the model based on the specified action.

    Parameters
    ----------
    model : object
        The model to be loaded or saved.
    model_path : str
        The file path where the model will be saved or loaded from.
    action : str, optional
        The action to perform ('load' or 'save'). Default is 'load'.

    Returns
    -------
    object
        The loaded model if action is 'load'.
    """
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
    """
    Loads or saves the parameters based on the specified action.

    Parameters
    ----------
    params : dict
        The parameters to be loaded or saved.
    params_path : str
        The file path where the parameters will be saved or loaded from.
    action : str, optional
        The action to perform ('load' or 'save'). Default is 'load'.

    Returns
    -------
    dict
        The loaded parameters if action is 'load'.
    """
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
    """
    Trains the model using the provided training data and evaluates it on the test data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.
    X_test : pandas.DataFrame
        The test data features.
    y_test : pandas.Series
        The test data labels.
    model_path : str
        The file path where the model will be saved.
    params_path : str
        The file path where the parameters will be saved.

    Returns
    -------
    tuple
        The trained ensemble model and the test accuracy.
    """
    try:
        ensemble_model = load_or_save_model(None, model_path, action='load')
        best_params = load_or_save_params(None, params_path, action='load')
    except Exception as e:
        logging.error(f"Error loading model or parameters: {e}")
        ensemble_model = None
        best_params = None


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
    target_names = ['Safe', 'Not Safe']


    # Print the performance metrics
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"Classification Report for Training Data:\n{classification_report(y_train, y_train_pred, target_names=target_names)}")
    print(f"\nClassification Report for Test Data:\n{classification_report(y_test, y_test_pred, target_names=target_names)}")

    return ensemble_model, test_accuracy



def conduct_optuna_study(X_train, y_train):
    """
    Conducts an Optuna study to find the best hyperparameters for the models.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.

    Returns
    -------
    dict
        The best hyperparameters for each model.
    """
    best_params = {}


    # Optimize XGBoost parameters
    def xgb_objective(trial):
        n_estimators_xgb = trial.suggest_int('n_estimators_xgb', 50, 100)
        max_depth_xgb = trial.suggest_int('max_depth_xgb', 3, 10)
        learning_rate_xgb = trial.suggest_float('learning_rate_xgb', 0.01, 0.3)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)  # Increase the range for stronger L1 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)  # Increase the range for stronger L2 regularization

        model = XGBClassifier(
            n_estimators=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=learning_rate_xgb,
            reg_alpha=reg_alpha,  # L1
            reg_lambda=reg_lambda,  # L2
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        return accuracy_score(y_train, y_train_pred)


    # Optimize XGBoost parameters
    xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    xgb_study.optimize(xgb_objective, n_trials=5)
    best_params['xgb'] = xgb_study.best_params
    def svm_objective(trial):
        try:
            C_svm = trial.suggest_float('C_svm', 0.1, 1.0)  # Regularization parameter for SVM
            kernel_svm = trial.suggest_categorical('kernel_svm', ['linear', 'rbf', 'poly'])

            model = SVC(
                C=C_svm,
                kernel=kernel_svm,
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)
        except Exception as e:
            logging.error(f"Error in SVM objective function: {e}")
            return 0  # Return a low score if there's an error


    # Optimize SVM parameters
    svm_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    svm_study.optimize(svm_objective, n_trials=5)
    best_params['svm'] = svm_study.best_params
    def logreg_objective(trial):
        try:
            C_logreg = trial.suggest_float('C_logreg', 0.0001, 1.0)  # Set a lower upper bound for stronger regularization
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])  # Regularization type

            model = LogisticRegression(
                C=C_logreg,
                penalty=penalty,
                solver='saga' if penalty == 'l1' else 'lbfgs',  # saga for L1, lbfgs for L2
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)
        except Exception as e:
            logging.error(f"Error in Logistic Regression objective function: {e}")
            return 0  # Return a low score if there's an error

    # Optimize Logistic Regression parameters
    logreg_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    logreg_study.optimize(logreg_objective, n_trials=5)
    best_params['logreg'] = logreg_study.best_params

    return best_params



def load_optuna_model(path):
    """
    Loads the Optuna model from the specified path.

    Parameters
    ----------
    path : str
        The file path where the model is saved.

    Returns
    -------
    object
        The loaded Optuna model.
    """
    return joblib.load(path)



def train_ensemble_model(best_params, X_train, y_train, model_path):
    """
    Trains an ensemble model using the best hyperparameters.

    This function trains a stacking ensemble model consisting of a Bagged SVM and an XGBoost model as base models,
    with a Logistic Regression model as the meta-model. The best hyperparameters for each model are provided
    through the `best_params` dictionary. The trained model is saved to the specified file path.

    Args:
        best_params (dict): The best hyperparameters for each model.
            - 'xgb': Hyperparameters for the XGBoost model.
                - 'n_estimators_xgb' (int): Number of boosting rounds.
                - 'max_depth_xgb' (int): Maximum tree depth for base learners.
                - 'learning_rate_xgb' (float): Boosting learning rate.
                - 'reg_alpha' (float, optional): L1 regularization term on weights (default is 0.0).
                - 'reg_lambda' (float, optional): L2 regularization term on weights (default is 1.0).
            - 'svm': Hyperparameters for the SVM model.
                - 'C_svm' (float): Regularization parameter.
                - 'kernel_svm' (str): Specifies the kernel type to be used in the algorithm.
            - 'logreg': Hyperparameters for the Logistic Regression model.
                - 'C_logreg' (float): Inverse of regularization strength (smaller values specify stronger regularization).
                - 'penalty' (str, optional): Used to specify the norm used in the penalization ('l1' or 'l2', default is 'l2').
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels.
        model_path (str): The file path where the model will be saved.

    Returns:
        object: The trained ensemble model.
    """
    logging.info(f"Training new ensemble model with best parameters")

    # XGBoost model with increased L1 and L2 regularization
    xgb_model = XGBClassifier(
        n_estimators=best_params['xgb']['n_estimators_xgb'],
        max_depth=best_params['xgb']['max_depth_xgb'],
        learning_rate=best_params['xgb']['learning_rate_xgb'],
        reg_alpha=best_params['xgb'].get('reg_alpha', 0.0),  # L1 regularization (default 0.0)
        reg_lambda=best_params['xgb'].get('reg_lambda', 1.0),  # L2 regularization (default 1.0)
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Adjust for class imbalance
        random_state=42,
        n_jobs=2
    )

    # Bagged SVM Model
    bagged_svm = BaggingClassifier(
        estimator=SVC(
            C=best_params['svm']['C_svm'],  # Regularization strength for SVM (higher C = less regularization)
            kernel=best_params['svm']['kernel_svm'],
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        n_estimators=10,  # Number of bagged models
        n_jobs=2,
        random_state=42
    )

    # Logistic Regression with increased regularization strength
    penalty = best_params['logreg'].get('penalty', 'l2')  # L1 or L2 penalty
    solver = 'saga' if penalty == 'l1' else 'lbfgs'  # Use 'saga' for L1, 'lbfgs' for L2

    # Stronger regularization by reducing the C parameter (higher C = weaker regularization)
    meta_model = LogisticRegression(
        C=best_params['logreg']['C_logreg'],  # Regularization strength (smaller C = stronger regularization)
        penalty=penalty,
        class_weight='balanced',
        random_state=42,
        solver=solver,
        max_iter=2000
    )

    # Stacking ensemble with Bagged SVM and XGBoost as base models
    stacking_model = StackingClassifier(
        estimators=[('bagged_svm', bagged_svm), ('xgb', xgb_model)],
        final_estimator=meta_model
    )

    # Train the ensemble model
    for _ in tqdm(range(1), desc="Training ensemble model"):
        stacking_model.fit(X_train, y_train)

    # Save the ensemble model
    joblib.dump(stacking_model, model_path)
    logging.info(f"Ensemble model trained and saved to {model_path}.\n")

    return stacking_model



def log_label_percentages(df, dataset_name):
    """
    Logs the percentage of each label in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    dataset_name : str
        The name of the dataset being processed.

    Returns
    -------
    None
    """
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
    """
    Counts the number of URLs in the provided list.

    Parameters
    ----------
    urls_list : list
        The list of URLs.

    Returns
    -------
    int
        The number of URLs in the list.
    """
    if isinstance(urls_list, list):
        return len(urls_list)
    else:
        return 0



def check_missing_values(df, df_name, num_rows=1):
    """
    Checks for missing values in the DataFrame and logs the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for missing values.
    df_name : str
        The name of the DataFrame.
    num_rows : int, optional
        The number of rows to display with missing values. Default is 1.

    Returns
    -------
    None
    """
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


# Redundant function
def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=6, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 6)):
    """
    Plots the learning curve for the provided estimator.

    Args:
        estimator (object): The estimator to plot the learning curve for.
        X (pd.DataFrame): The training data features.
        y (pd.Series): The training data labels.
        title (str): The title of the plot.
        ylim (tuple): The y-axis limits for the plot.
        cv (int): The number of cross-validation folds.
        n_jobs (int): The number of jobs to run in parallel.
        train_sizes (array): The sizes of the training sets.

    Returns:
        None
    """
    logging.info("Starting the plot_learning_curve function.")
    


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    


    try:
        logging.info("Calling sklearn's learning_curve function.")
        train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)
        logging.info("learning_curve function executed successfully.")
        

        # Plot the learning curves
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', color='g', label='Validation score')
        plt.legend(loc='best')
        plt.grid()
        logging.info("Plotting the learning curve.")
        plt.show()
        logging.info("Learning curve plot displayed successfully.\n")
    except Exception as e:
        logging.error(f"An error occurred while plotting the learning curve: {e}")



def get_fold_paths(fold_idx, base_dir='Processed Data'):
    """
    Generates file paths for the train and test data and labels for the specified fold.

    Parameters
    ----------
    fold_idx : int
        The index of the fold.
    base_dir : str, optional
        The base directory where the data will be saved. Default is 'Processed Data'.

    Returns
    -------
    tuple
        The file paths for the train data, test data, train labels, test labels, and preprocessor.
    """
    train_data_path = os.path.join(base_dir, f"fold_{fold_idx}_train_data.npz")
    test_data_path = os.path.join(base_dir, f"fold_{fold_idx}_test_data.npz")
    train_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_train_labels.pkl")
    test_labels_path = os.path.join(base_dir, f"fold_{fold_idx}_test_labels.pkl")
    preprocessor_path = os.path.join(base_dir, f"fold_{fold_idx}_preprocessor.pkl")
    
    return train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path



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
    train_data_path, test_data_path, train_labels_path, test_labels_path, preprocessor_path = get_fold_paths(fold_idx, dir)

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
        X_train_non_text_processed = preprocessor.fit_transform(X_train_non_text)
        X_test_non_text_processed = preprocessor.transform(X_test_non_text)
        feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
        logging.info(f"Columns in X_train after processing non-text features: {X_train_non_text_processed.shape}")
        logging.info(f"Feature names: {feature_names}")
        if X_train_non_text_processed.shape[0] != y_train.shape[0]:
            logging.error(f"Row mismatch: {X_train_non_text_processed.shape[0]} vs {y_train.shape[0]}")
        logging.info(f"Non text features processed for fold {fold_idx}.\n")


        # Save the preprocessor
        logging.info(f"Saving preprocessor for fold {fold_idx}...")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved preprocessor to {preprocessor_path}\n")


        # Transform the text features
        logging.info(f"Extracting BERT features for X_train for {fold_idx}...")
        X_train_text_processed = pipeline.named_steps['bert_features'].transform(X_train['cleaned_text'].tolist())
        logging.info(f"Extracting BERT features for X_test for {fold_idx}...")
        X_test_text_processed = pipeline.named_steps['bert_features'].transform(X_test['cleaned_text'].tolist())
        logging.info(f"Number of features extracted from BERT for fold {fold_idx}: {X_train_text_processed.shape}")
        logging.info(f"Bert features extracted for fold {fold_idx}.\n")


        # Combine processed features
        logging.info(f"Combining processed features for fold {fold_idx}...")
        X_train_combined = np.hstack([X_train_non_text_processed, X_train_text_processed])
        X_test_combined = np.hstack([X_test_non_text_processed, X_test_text_processed])
        logging.info(f"Total number of combined features for fold {fold_idx}: {X_train_combined.shape}")
        logging.info(f"Combined processed features for fold {fold_idx}.\n")



        logging.info(f"Class distribution before SMOTE for fold {fold_idx}: {Counter(y_train)}")
        logging.info(f"Applying SMOTE to balance the training data for fold {fold_idx}...")
        X_train_balanced, y_train_balanced = pipeline.named_steps['smote'].fit_resample(X_train_combined, y_train)
        logging.info(f"Class distribution after SMOTE for fold {fold_idx}: {Counter(y_train_balanced)}")
        logging.info(f"SMOTE applied for fold {fold_idx}.\n")



        logging.info(f"Applying PCA for dimensionality reduction for fold {fold_idx}...")
        X_train_balanced = pipeline.named_steps['pca'].fit_transform(X_train_balanced)
        X_test_combined = pipeline.named_steps['pca'].transform(X_test_combined)


        # Log the number of features after PCA
        n_components = pipeline.named_steps['pca'].n_components_
        logging.info(f"Number of components after PCA: {n_components}")
        logging.info(f"Shape of X_train after PCA: {X_train_balanced.shape}")


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



def extract_email(text):
    """
    Extract the email address from a given text.

    Parameters
    ----------
    text : str
        The text containing the email address.

    Returns
    -------
    str or None
        The extracted email address or None if no email address is found.
    """
    if isinstance(text, str):
        match = re.search(r'<([^>]+)>', text)
        if match:
            return match.group(1)
        elif re.match(r'^[^@]+@[^@]+\.[^@]+$', text):
            return text
    return None



def process_and_save_emails(df, output_file):
    """
    Process the DataFrame to extract sender and receiver emails and save to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    output_file : str
        The file path to save the processed email data.

    Returns
    -------
    pandas.DataFrame
        The DataFrame containing the extracted emails.
    """
    # Extract sender and receiver emails
    df['sender'] = df['sender'].apply(extract_email)
    df['receiver'] = df['receiver'].apply(extract_email)
    
    # Create a new DataFrame with the extracted emails
    email_df = df[['sender', 'receiver']]
    
    # Save the new DataFrame to a CSV file
    email_df.to_csv(output_file, index=False)
    return email_df



def load_or_save_emails(df, output_file, df_name = 'CEAS_08'):
    """
    Load the cleaned email data from a CSV file or process and save the data if the file does not exist.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    output_file : str
        The file path to save or load the cleaned email data.
    df_name : str, optional
        The name of the DataFrame source. Default is 'CEAS_08'.

    Returns
    -------
    pandas.DataFrame
        The cleaned email data.
    """
    if os.path.exists(output_file):
        logging.info(f"Output file {output_file} already exists. Loading data from {output_file}...\n")
        df_cleaned = pd.read_csv(output_file)
    else:
        logging.info(f"Output file {output_file} does not exist. Loading data from {df_name}...")
        logging.info(f"Data loaded from {df_name}. Beginning processing...")
        
        df_cleaned = process_and_save_emails(df, output_file)
        
        logging.info(f"Data processing completed. Cleaned data saved to {output_file}.")
    
    return df_cleaned



def generate_noisy_dataframe(data, file_path, noise_level=0.1):
    """
    Generates a noisy DataFrame by injecting noise into specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The original DataFrame to be processed.
    file_path : str
        The path to save or load the noisy DataFrame.
    noise_level : float
        The level of noise to inject (probability for categorical/text, standard deviation for numerical).

    Returns
    -------
    pandas.DataFrame
        The noisy DataFrame.
    """
    
    # Function to inject noise into numerical columns
    def inject_numerical_noise(data, columns, noise_level):
        for column in columns:
            noise = np.random.normal(0, noise_level, data[column].shape)
            data[column] += noise
        return data

    # Function to inject noise into text columns
    def inject_text_noise(data, text_column, noise_level):
        for i in range(len(data)):
            if random.random() < noise_level:  # With a probability of noise_level
                text_list = list(data[text_column][i])
                pos = random.randint(0, len(text_list) - 1)  # Choose a random position
                text_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')  # Replace with a random letter
                data.at[i, text_column] = ''.join(text_list)
        return data

    # Function to inject noise into categorical columns
    def inject_categorical_noise(data, columns, noise_level):
        for column in columns:
            unique_values = data[column].unique()
            for i in range(len(data)):
                if random.random() < noise_level:  # With a probability of noise_level
                    data.at[i, column] = random.choice(unique_values)
        return data

    # Check if the noisy DataFrame already exists
    if os.path.exists(file_path):
        logging.info(f"Noisy DataFrame already exists as '{file_path}'. Loading it.")
        df_noisy = pd.read_csv(file_path)
    else:
        logging.info(f"'{file_path}' does not exist. Generating a noisy DataFrame.")
        
        # Define the columns for noise injection
        numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
        categorical_columns = ['sender', 'receiver']
        
        # Apply noise injection
        data = inject_numerical_noise(data, numerical_columns, noise_level)
        data = inject_text_noise(data, 'cleaned_text', noise_level)
        data = inject_categorical_noise(data, categorical_columns, noise_level)

        # Save the noisy DataFrame
        data.to_csv(file_path, index=False)
        logging.info(f"Noisy DataFrame saved as '{file_path}'.")

    return data



# Main processing function
def main():
    """
    Load configuration settings from a JSON file and set up file paths for various data processing stages.

    This code snippet performs the following tasks:
    1. Loads configuration settings from 'config.json'.
    2. Sets up base directory and file paths for different stages of data processing, including:
    - Dataset file
    - Preprocessed files
    - Extracted email header files
    - Merged files
    - Cleaned files

    Attributes:
        config (dict): Configuration settings loaded from 'config.json'.
        base_dir (str): Base directory for data files.
        dataset (str): Path to the CEAS_08 dataset file.
        PreprocessedSpamAssassinFile (str): Path to the preprocessed SpamAssassin file.
        PreprocessedCEASFile (str): Path to the preprocessed CEAS_08 file.
        ExtractedSpamAssassinEmailHeaderFile (str): Path to the extracted SpamAssassin email header file.
        ExtractedCEASEmailHeaderFile (str): Path to the extracted CEAS email header file.
        MergedSpamAssassinFile (str): Path to the merged SpamAssassin file.
        MergedCEASFile (str): Path to the merged CEAS_08 file.
        MergedDataFrame (str): Path to the merged DataFrame file.
        CleanedDataFrame (str): Path to the cleaned DataFrame file.
        CleanedCEASHeaders (str): Path to the cleaned CEAS headers file.
        MergedCleanedCEASHeaders (str): Path to the merged cleaned CEAS headers file.
        MergedCleanedDataFrame (str): Path to the merged cleaned DataFrame file.
    """
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    base_dir = config['base_dir']
    CEAS_08_Dataset = os.path.join(base_dir, 'CEAS_08.csv')
    PreprocessedSpamAssassinFile = os.path.join(base_dir, 'Data Preprocessing', 'PreprocessedSpamAssassin.csv')
    PreprocessedCEASFile = os.path.join(base_dir, 'Data Preprocessing', 'PreprocessedCEAS_08.csv')
    ExtractedSpamAssassinEmailHeaderFile = os.path.join(base_dir, 'Feature Engineering', 'SpamAssassinExtractedEmailHeader.csv')
    ExtractedCEASEmailHeaderFile = os.path.join(base_dir, 'Feature Engineering', 'CEASExtractedEmailHeader.csv')
    MergedSpamAssassinFile = os.path.join(base_dir, 'Data Integration', 'MergedSpamAssassin.csv')
    MergedCEASFile = os.path.join(base_dir, 'Data Integration', 'MergedCEAS_08.csv')
    MergedDataFrame = os.path.join(base_dir, 'Data Integration', 'MergedDataFrame.csv')
    CleanedDataFrame = os.path.join(base_dir, 'Data Cleaning', 'CleanedDataFrame.csv')
    CleanedCEASHeaders = os.path.join(base_dir, 'Data Cleaning', 'CleanedCEASHeaders.csv')
    MergedCleanedCEASHeaders = os.path.join(base_dir, 'Data Cleaning', 'MergedCleanedCEASHeaders.csv')
    MergedCleanedDataFrame = os.path.join(base_dir, 'Data Cleaning', 'MergedCleanedDataFrame.csv')
    NoisyDataFrame = os.path.join(base_dir, 'Noise Injection', 'NoisyDataFrame.csv')
    pipeline_path = os.path.join(base_dir, 'Feature Extraction')


    # Load the datasets
    df_ceas = pd.read_csv(CEAS_08_Dataset, sep=',', encoding='utf-8')
    dataset = load_dataset('talby/spamassassin',split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()



    try:
        # ****************************** #
        #       Data Preprocessing       #
        # ****************************** #
        """
        Perform data preprocessing for the SpamAssassin and CEAS_08 datasets.

        This code snippet performs the following tasks:
        1. Logs the beginning of the data preprocessing stage.
        2. Changes label values in the SpamAssassin dataset to match the labeling scheme.
        3. Removes duplicates and missing values from both datasets using the DatasetProcessor class.
        4. Combines the processed SpamAssassin and CEAS_08 datasets into a single DataFrame.
        5. Logs label percentages for the individual and combined datasets.
        6. Checks for missing values in the combined DataFrame.
        7. Logs the completion of the data preprocessing stage.

        Attributes:
            df_spamassassin (pandas.DataFrame): The SpamAssassin dataset.
            df_ceas (pandas.DataFrame): The CEAS_08 dataset.
            processor_spamassassin (DatasetProcessor): Processor for the SpamAssassin dataset.
            df_processed_spamassassin (pandas.DataFrame): Processed SpamAssassin dataset.
            processor_ceas (DatasetProcessor): Processor for the CEAS_08 dataset.
            df_processed_ceas (pandas.DataFrame): Processed CEAS_08 dataset.
            combined_percentage_df (pandas.DataFrame): Combined DataFrame of processed SpamAssassin and CEAS_08 datasets.
        """

        logging.info(f"Beginning Data Preprocessing...")


        # Change label values to match the labeling scheme
        df_spamassassin['label'] = df_spamassassin['label'].map({1: 0, 0: 1})


        # Remove duplicates and missing values
        processor_spamassassin = DatasetProcessor(df_spamassassin, 'text', 'SpamAssassin', PreprocessedSpamAssassinFile)
        df_processed_spamassassin = processor_spamassassin.process_dataset()
        processor_ceas = DatasetProcessor(df_ceas, 'body', 'CEAS_08', PreprocessedCEASFile)
        df_processed_ceas = processor_ceas.process_dataset()


        # Combined DataFrame
        combined_percentage_df = pd.concat([df_processed_spamassassin, df_processed_ceas])


        # Check if DataFrame has merged correctly
        log_label_percentages(df_processed_ceas, 'CEAS_08')
        log_label_percentages(df_processed_spamassassin, 'SpamAssassin')
        log_label_percentages(combined_percentage_df,'Combined CEAS_08 and SpamAssassin (No Processing)')
        check_missing_values(combined_percentage_df, 'Combined CEAS_08 and SpamAssassin (No Processing)')
        logging.info(f"Data Preprocessing completed.\n")
        #Columns in CEAS_08 dataset: ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
        #Columns in SpamAssassin dataset: ['label', 'group', 'text']
        

        # ****************************** #
        #       Feature Engineering      #
        # ****************************** #
        """
        Perform feature engineering on the processed datasets.

        This code snippet performs the following tasks:
        1. Logs the beginning of the feature engineering stage.
        2. Extracts email headers from the SpamAssassin dataset using the EmailHeaderExtractor.
        3. Logs the completion of email header extraction and saving for the SpamAssassin dataset.
        4. Converts text URLs to numerical counts in the SpamAssassin dataset.
        5. Drops unnecessary columns ('mailto' and 'texturls') from the SpamAssassin dataset.
        6. Extracts email headers from the CEAS_08 dataset using the EmailHeaderExtractor.
        7. Logs the completion of email header extraction and saving for the CEAS_08 dataset.
        8. Logs the completion of the feature engineering stage.

        Attributes:
            spamassassin_headers_df (pandas.DataFrame): DataFrame containing extracted email headers from the SpamAssassin dataset.
            ceas_headers_df (pandas.DataFrame): DataFrame containing extracted email headers from the CEAS_08 dataset.
        """

        logging.info(f"Beginning Feature Engineering...")


        # Extract email headers from the SpamAssassin dataset
        spamassassin_headers_df = load_or_extract_headers(df_processed_spamassassin, ExtractedSpamAssassinEmailHeaderFile, EmailHeaderExtractor, 'Spam Assassin')
        logging.info("Email header extraction and saving from Spam Assassin completed.\n")
        # Columns in current spam assassin email headers: ['sender', 'receiver', 'mailto', 'texturls', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address']
        spamassassin_headers_df['urls'] = spamassassin_headers_df['texturls'].apply(count_urls) # Convert text to number for URLs
        # Columns in current spam assassin email headers: ['sender', 'receiver', 'mailto', 'texturls', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls']
        spamassassin_headers_df.drop(columns=['mailto'], inplace=True) # Drop the 'mailto' column
        spamassassin_headers_df.drop(columns=['texturls'], inplace=True) # Drop the 'texturls' column
        # Columns in current spam assassin email headers: ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls']
        ceas_headers_df = load_or_extract_headers(df_processed_ceas, ExtractedCEASEmailHeaderFile, EmailHeaderExtractor, 'CEAS_08')
        # Columns in current ceas email headers: ['https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address']
        logging.info("Email header extraction and saving from CEAS completed.")
        logging.info(f"Feature Engineering completed.\n")



        # ************************* #
        #       Data Cleaning       #
        # ************************* #
        """
        Perform data cleaning on the CEAS_08 dataset, specifically focusing on the 'sender' and 'receiver' columns.

        This code snippet performs the following tasks:
        1. Logs the beginning of the data cleaning stage for the 'sender' and 'receiver' columns in the CEAS_08 dataset.
        2. Loads or saves the cleaned email headers from the CEAS_08 dataset.
        3. Logs the beginning of the merging process for the cleaned headers with the processed CEAS_08 dataset.
        4. Checks if the number of rows in the cleaned headers matches the number of rows in the processed CEAS_08 dataset.
        5. Drops the 'sender' and 'receiver' columns from the processed CEAS_08 dataset.
        6. Merges the cleaned headers with the processed CEAS_08 dataset.
        7. Logs the number of missing rows in the merged DataFrame.
        8. Logs the total number of rows in the processed CEAS_08 dataset and the merged DataFrame.
        9. Checks if the number of rows in the merged DataFrame matches the number of rows in the processed CEAS_08 dataset.
        10. Saves the merged DataFrame to a CSV file if the row counts match.
        11. Logs the completion of the data cleaning stage for the 'sender' and 'receiver' columns in the CEAS_08 dataset.

        Attributes:
            df_cleaned_ceas_headers (pandas.DataFrame): DataFrame containing cleaned email headers from the CEAS_08 dataset.
            df_cleaned_ceas_headers_merge (pandas.DataFrame): DataFrame containing the merged cleaned headers and processed CEAS_08 dataset.
            missing_in_cleaned_ceas_header_merged (pandas.DataFrame): DataFrame containing rows with missing 'sender' or 'receiver' values in the merged DataFrame.
        """
        
        logging.info(f"Beginning Data Cleaning of CEAS_08 ['sender', 'receiver']...")
        df_cleaned_ceas_headers = load_or_save_emails(df_processed_ceas, CleanedCEASHeaders)
        # Columns in cleaned ceas email headers: ['sender', 'receiver']
        


        logging.info(f"Begining merging of Cleaned Headers of CEAS_08 with Processed CEAS_08...")
        if len(df_cleaned_ceas_headers) != len(df_processed_ceas):
            logging.error("The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
            raise ValueError("The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
        else:
            df_processed_ceas.drop(columns=['sender', 'receiver'], inplace=True)
            logging.info(f"Columns in df_cleaned_ceas_headers: {df_cleaned_ceas_headers.columns.tolist()}")
            logging.info(f"Columns in df_processed_ceas: {df_processed_ceas.columns.tolist()}")
            df_cleaned_ceas_headers_merge = pd.concat([df_cleaned_ceas_headers.reset_index(drop=True), df_processed_ceas.reset_index(drop=True)], axis=1)
            #df_cleaned_ceas_headers_merge.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
            missing_in_cleaned_ceas_header_merged = df_cleaned_ceas_headers_merge[(df_cleaned_ceas_headers_merge['sender'].isnull()) | (df_cleaned_ceas_headers_merge['receiver'].isnull())]
            logging.info(f"Number of missing rows in Merged Cleaned Headers of CEAS_08 DataFrame: {len(missing_in_cleaned_ceas_header_merged)}")
            logging.info(f'Total rows in Processed CEAS_08 Dataframe: {len(df_processed_ceas)}')
            logging.info(f"Total rows in Merged Cleaned Headers of CEAS_08 Dataframe: {len(df_cleaned_ceas_headers_merge)}")
        if len(df_cleaned_ceas_headers_merge) != len(df_processed_ceas):
            logging.error("The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.")
            raise ValueError("The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame does not match Processed CEAS_08.\n")
        else:
            logging.info("The number of rows in the Merged Cleaned Headers of CEAS_08 DataFrame matches Processed CEAS_08.")
            df_cleaned_ceas_headers_merge.to_csv(MergedCleanedCEASHeaders, index=False)
            logging.info(f"Merged Cleaned Headers of CEAS_08 DataFrame successfully saved to {MergedCleanedCEASHeaders}")
        logging.info(f"Data Cleaning of CEAS_08 ['sender', 'receiver'] completed.\n")



        # ****************************** #
        #       Data Integration         #
        # ****************************** #
        """
        Perform data integration by merging processed datasets and extracted information.

        This code snippet performs the following tasks:
        1. Logs the beginning of the data integration stage.
        2. Merges the processed SpamAssassin dataset with the extracted email headers.
        3. Verifies the merged SpamAssassin DataFrame.
        4. Saves the merged SpamAssassin DataFrame to a CSV file.
        5. Merges the processed CEAS_08 dataset with the extracted email headers.
        6. Verifies the merged CEAS_08 DataFrame.
        7. Saves the merged CEAS_08 DataFrame to a CSV file.
        8. Merges the SpamAssassin and CEAS_08 datasets into a combined DataFrame.
        9. Verifies the combined DataFrame for label consistency and row count.
        10. Saves the combined DataFrame to a CSV file.

        Attributes:
            df_processed_spamassassin (pandas.DataFrame): Processed SpamAssassin dataset.
            spamassassin_headers_df (pandas.DataFrame): Extracted email headers from the SpamAssassin dataset.
            merged_spamassassin_df (pandas.DataFrame): Merged DataFrame of processed SpamAssassin dataset and extracted email headers.
            df_cleaned_ceas_headers_merge (pandas.DataFrame): Merged cleaned headers and processed CEAS_08 dataset.
            ceas_headers_df (pandas.DataFrame): Extracted email headers from the CEAS_08 dataset.
            merged_ceas_df (pandas.DataFrame): Merged DataFrame of processed CEAS_08 dataset and extracted email headers.
            combined_df (pandas.DataFrame): Combined DataFrame of merged SpamAssassin and CEAS_08 datasets.
            combined_percentage_df (pandas.DataFrame): Combined DataFrame of processed SpamAssassin and CEAS_08 datasets without additional processing.
        """

        logging.info(f"Beginning Data Integration...")


        # Merging Processed SpamAssassin dataset with the extracted information
        logging.info(f"Merging Processed Spam Assassin and Spam Assassin Header Dataframes...")
        df_processed_spamassassin.reset_index(inplace=True)
        spamassassin_headers_df.reset_index(inplace=True)
        #spamassassin_headers_df.fillna({'sender': 'unknown', 'receiver': 'unknown'}, inplace=True)
        if len(df_processed_spamassassin) == len(spamassassin_headers_df):
            merged_spamassassin_df = pd.merge(df_processed_spamassassin, spamassassin_headers_df, on='index', how='left')
            merged_spamassassin_df = merged_spamassassin_df.rename(columns={'text': 'body'})
            merged_spamassassin_df = merged_spamassassin_df[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']]
            missing_in_merged_df = merged_spamassassin_df[merged_spamassassin_df['index'].isnull()]
            logging.info(f"Number of missing rows in Merged Spam Assassin Dataframe: {len(missing_in_merged_df)}")
            logging.info(f'Total rows in Processed Spam Assassin Dataframe: {len(df_processed_spamassassin)}')
            logging.info(f"Total rows in Merged Spam Assassin Dataframe: {len(merged_spamassassin_df)}")
            merged_spamassassin_df.drop(columns=['index'], inplace=True)
            # Columns in merged_spamassassin_df: ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label']
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
            logging.info(f"Merged Spam Assassin DataFrame successfully saved to {MergedSpamAssassinFile}\n")
    

        # Merge Processed CEAS_08 dataset with the extracted information
        logging.info(f"Merging Processed CEAS_08 and CEAS_08 Header Dataframes...")
        df_cleaned_ceas_headers_merge.reset_index(inplace=True)
        ceas_headers_df.reset_index(inplace=True)
        if len(df_processed_spamassassin) == len(spamassassin_headers_df):
            merged_ceas_df = pd.merge(df_cleaned_ceas_headers_merge, ceas_headers_df, on='index', how='left')
            merged_ceas_df = merged_ceas_df[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label', 'index']]
            missing_in_merged_df = merged_ceas_df[merged_ceas_df['index'].isnull()]
            logging.info(f"Number of missing rows in Merged CEAS_08 Dataframe: {len(missing_in_merged_df)}")
            logging.info(f'Total rows in Processed CEAS_08 Dataframe: {len(df_cleaned_ceas_headers_merge)}')
            logging.info(f"Total rows in Merged CEAS_08 Dataframe: {len(merged_ceas_df)}")
            merged_ceas_df.drop(columns=['index'], inplace=True)
            # Columns in merged_ceas_df: ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label']
        else:
            logging.error("The number of rows in Processed CEAS_08 and CEAS_08 Header do not match.")
            raise ValueError("The number of rows in Processed CEAS_08 and CEAS_08 Header do not match.")
        

        # Verifying the merged CEAS_08 DataFrame
        if len(merged_ceas_df) != len(df_cleaned_ceas_headers_merge):
            logging.error("The number of rows in the Merged CEAS_08 DataFrame DataFrame does not match Processed CEAS_08.")
            raise ValueError("The number of rows in the Merged CEAS_08 DataFrame does not match Processed CEAS_08.")
        else:
            logging.info(f"The number of rows in the Merged CEAS_08 DataFrame matches Processed CEAS_08.")
            merged_ceas_df.to_csv(MergedCEASFile, index=False)
            logging.info(f"Merged CEAS_08 DataFrame successfully saved to {MergedCEASFile}\n")
        

        # Merge Spam Assassin and CEAS_08 datasets
        logging.info(f"Merging Spam Assassin and CEAS_08 Dataframes...")
        common_columns = ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label']
        df_spamassassin_common = merged_spamassassin_df[common_columns]
        df_ceas_common = merged_ceas_df[common_columns]
        combined_df = pd.concat([df_spamassassin_common, df_ceas_common])


        # Verifying the combined DataFrame
        combined_labels = set(combined_df['label'].unique())
        percentage_labels = set(combined_percentage_df['label'].unique())
        if combined_labels != percentage_labels:
            logging.error(f"Labels in Merged DataFrame do not match those in Combined CEAS_08 and SpamAssassin (No Processing). "
                        f"Merged DataFrame labels: {combined_labels}, "
                        f"Combined Processed DataFrame labels: {percentage_labels}")
            raise ValueError("Labels do not match between Merged DataFrame and Combined CEAS_08 and SpamAssassin (No Processing).")
        else:
            logging.info("Labels in Merged DataFrame match those in Combined CEAS_08 and SpamAssassin (No Processing).")
        combined_label_counts = combined_df['label'].value_counts().sort_index()
        percentage_label_counts = combined_percentage_df['label'].value_counts().sort_index()
        if not combined_label_counts.equals(percentage_label_counts):
            logging.error("Label distributions in Merged DataFrame do not match those in Combined CEAS_08 and SpamAssassin (No Processing).")
            logging.error(f"Merged DataFrame distributions:\n{combined_label_counts}")
            logging.error(f"Combined CEAS_08 and SpamAssassin (No Processing) distributions:\n{percentage_label_counts}")
            raise ValueError("Label distributions do not match between Merged DataFrame and Combined CEAS_08 and SpamAssassin (No Processing).")
        else:
            logging.info("Label distributions in Merged DataFrame match those in Combined CEAS_08 and SpamAssassin (No Processing).")

        if len(combined_df) != len(combined_percentage_df):
            logging.error("The number of rows in the Merged DataFrame does not match the Combined CEAS_08 and SpamAssassin (No Processing).")
            raise ValueError("The number of rows in the Merged DataFrame does not match the Combined CEAS_08 and SpamAssassin (No Processing).")
        else:
            logging.info("The number of rows in the Merged DataFrame matches the Combined CEAS_08 and SpamAssassin (No Processing).")


        # Save the Merged DataFrame
        combined_df.to_csv(MergedDataFrame, index=False)
        # Columns in Merged DataFrame: ['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'body', 'label']
        logging.info(f"Merged DataFrame successfully saved to {MergedDataFrame}")
        logging.info(f"Data Integration completed.\n")


        # ************************* #
        #       Data Cleaning       #
        # ************************* #
        """
        Perform data cleaning on the 'body' column of the combined DataFrame and integrate the cleaned data.

        This code snippet performs the following tasks:
        1. Logs the beginning of the data cleaning stage for the 'body' column.
        2. Loads or cleans the 'body' column data from the combined DataFrame.
        3. Concatenates the cleaned DataFrame with the merged DataFrame.
        4. Verifies the cleaned combined DataFrame for label consistency and row count.
        5. Saves the cleaned combined DataFrame to a CSV file.

        Attributes:
            df_clean_body (pandas.DataFrame): DataFrame containing the cleaned 'body' column data.
            combined_df_reset (pandas.DataFrame): Combined DataFrame with reset index.
            df_clean_body_reset (pandas.DataFrame): Cleaned 'body' column DataFrame with reset index.
            df_cleaned_combined (pandas.DataFrame): DataFrame containing the combined cleaned data.
            combined_labels (numpy.ndarray): Unique labels in the combined DataFrame.
            df_cleaned_combined_labels (numpy.ndarray): Unique labels in the cleaned combined DataFrame.
            combined_label_counts (pandas.Series): Label counts in the combined DataFrame.
            df_cleaned_combined_label_counts (pandas.Series): Label counts in the cleaned combined DataFrame.
        """

        logging.info(f"Beginning Data Cleaning ['body']...")
        df_clean_body = load_or_clean_data('Merged Dataframe', combined_df, 'body', CleanedDataFrame, data_cleaning)


        # Concatenate the Cleaned DataFrame with the Merged DataFrame
        logging.info(f"Combining Cleaned DataFrame with Merged DataFrame...")
        combined_df_reset = combined_df.reset_index(drop=True)
        df_clean_body_reset = df_clean_body.reset_index(drop=True)
        df_cleaned_combined = pd.concat([
            combined_df_reset[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'label']],  # Select necessary columns from merged
            df_clean_body_reset[['cleaned_text']]  # Select the cleaned_text and label from df_clean
        ], axis=1)
        logging.info(f"Dataframes combined successfully.\n")


        # Verifying the Cleaned Combine DataFrame
        logging.info(f"Verifying the Cleaned Combined DataFrame...")
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
        df_cleaned_combined = df_cleaned_combined[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count', 'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']]    
        df_cleaned_combined.to_csv(MergedCleanedDataFrame, index=False)
        logging.info(f"Data Cleaning completed.\n")

        # ***************************** #
        #       Noise Injection         #
        # ***************************** #
        """
        Perform noise injection on the cleaned combined DataFrame.

        This code snippet performs the following tasks:
        1. Logs the beginning of the noise injection stage.
        2. Generates a noisy DataFrame from the cleaned combined DataFrame.
        3. Logs the completion of the noise injection stage.

        Attributes:
            df_cleaned_combined (pandas.DataFrame): The cleaned combined DataFrame.
            NoisyDataFrame (str): File path to save the noisy DataFrame.
            noisy_df (pandas.DataFrame): DataFrame containing the noisy data.
        """
        logging.info(f"Beginning Noise Injection...")
        noisy_df = generate_noisy_dataframe(df_cleaned_combined, NoisyDataFrame)
        logging.info(f"Noise Injection completed.\n")

    
        # ************************* #
        #       Data Splitting      #
        # ************************* #
        """
        Perform data splitting using stratified k-fold cross-validation and initialize lists for storing accuracies and learning curve data.

        This code snippet performs the following tasks:
        1. Logs the beginning of the data splitting stage.
        2. Splits the cleaned combined DataFrame into stratified k-folds.
        3. Logs the completion of the data splitting stage.
        4. Initializes lists to store training accuracies, test accuracies, and learning curve data for each fold.

        Attributes:
            folds (list): List of stratified k-fold splits of the cleaned combined DataFrame.
            fold_train_accuracies (list): List to store training accuracies for each fold.
            fold_test_accuracies (list): List to store test accuracies for each fold.
            learning_curve_data (list): List to store learning curve data for each fold.
        """

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
            """
            Perform feature extraction, data augmentation, and preprocessing for a specific fold in a stratified k-fold cross-validation.

            This code snippet performs the following tasks:
            1. Defines columns for categorical, numerical, and text data.
            2. Initializes BERT feature extractor and transformer.
            3. Defines a preprocessor for categorical and numerical columns.
            4. Defines a pipeline with preprocessor, BERT, data augmentation, SMOTE, and PCA.
            5. Runs the pipeline or loads preprocessed data for the specified fold.
            6. Logs the successful processing or loading of data for the specified fold.

            Attributes:
                categorical_columns (list): List of categorical columns.
                numerical_columns (list): List of numerical columns.
                text_column (str): Name of the text column.
                bert_extractor (BERTFeatureExtractor): BERT feature extractor instance.
                bert_transformer (BERTFeatureTransformer): BERT feature transformer instance.
                preprocessor (ColumnTransformer): Preprocessor for categorical and numerical columns.
                pipeline (Pipeline): Pipeline with preprocessor, BERT, data augmentation, SMOTE, and PCA.
                X_train_balanced (pandas.DataFrame): Balanced training data.
                X_test_combined (pandas.DataFrame): Combined test data.
                y_train_balanced (pandas.Series): Balanced training labels.
                y_test (pandas.Series): Test labels.
            """

            logging.info(f"Beginning Feature Extraction for Fold {fold_idx}...")

            # Define columns for categorical, numerical, and text data
            categorical_columns = ['sender', 'receiver']
            numerical_columns = ['https_count', 'http_count', 'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
            text_column = 'cleaned_text'


            # Initialize BERT feature extractor and transformer
            bert_extractor = BERTFeatureExtractor()
            bert_transformer = BERTFeatureTransformer(feature_extractor=bert_extractor)


            # Define preprocessor for categorical and numerical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', Pipeline([
                        ('rare_cat_remover', RareCategoryRemover(threshold=0.05)),  # Remove rare categories
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


            
            # Define pipeline with preprocessor, BERT, and SMOTE
            pipeline = Pipeline(steps=[
                #('augment', DataAugmentationTransformer(categorical_columns=categorical_columns, numerical_columns=numerical_columns)),
                ('preprocessor', preprocessor),
                ('bert_features', bert_transformer),  # Custom transformer for BERT
                ('smote', SMOTE(random_state=42)),  # Apply SMOTE after augmentation
                ('pca', PCA(n_components=10))
            ])


            # Call the function to either run the pipeline or load preprocessed data
            X_train_balanced, X_test_combined, y_train_balanced, y_test = run_pipeline_or_load(
                fold_idx=fold_idx,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                pipeline=pipeline,
                dir=pipeline_path
            )
            logging.info(f"Data for Fold {fold_idx} has been processed or loaded successfully.\n")

            # ***************************************** #
            #       Model Training and Evaluation       #
            # ***************************************** #
            """
            Train the model and evaluate its performance for each fold in a stratified k-fold cross-validation.

            This code snippet performs the following tasks:
            1. Logs the beginning of the model training and evaluation stage for the specified fold.
            2. Defines the file paths for saving the trained model and best parameters.
            3. Trains the model and evaluates its performance for the specified fold.
            4. Appends the test accuracy for the fold to the list of fold test accuracies.
            5. Logs the successful processing, training, and evaluation of the model for the specified fold.
            6. Stores learning curve data for later plotting.

            Attributes:
                model_path (str): File path to save the trained model.
                params_path (str): File path to save the best parameters.
                ensemble_model (sklearn.base.BaseEstimator): Trained ensemble model.
                test_accuracy (float): Test accuracy for the current fold.
                fold_test_accuracies (list): List to store test accuracies for each fold.
                learning_curve_data (list): List to store learning curve data for each fold.
            """

            logging.info(f"Beginning Model Training and Evaluation for Fold {fold_idx}...")
            # Train the model and evaluate the performance for each fold
            model_path = os.path.join(base_dir, 'Models & Parameters', f'ensemble_model_fold_{fold_idx}.pkl')
            params_path = os.path.join(base_dir, 'Models & Parameters', f'best_params_fold_{fold_idx}.json')
            ensemble_model, test_accuracy = model_training(
                X_train_balanced,
                y_train_balanced,
                X_test_combined,
                y_test,
                model_path=model_path,
                params_path=params_path,
            )
            fold_test_accuracies.append(test_accuracy)
            logging.info(f"Data for Fold {fold_idx} has been processed, model trained, and evaluated.\n")


            # Store learning curve data for later plotting
            learning_curve_data.append((X_train_balanced, y_train_balanced, ensemble_model, fold_idx))


            # ********************************* #
            #       Plot Learning Curves        #
            # ********************************* #
            """
            Plot learning curves for each fold in the stratified k-fold cross-validation.

            This code snippet performs the following tasks:
            1. Iterates over the learning curve data for each fold.
            2. Plots the learning curve for the specified fold using the provided estimator, training data, and labels.
            3. Sets the title of the plot to indicate the fold number.
            4. Defines the training sizes and cross-validation splits for the learning curve plot.

            Attributes:
                learning_curve_data (list): List containing tuples of training data, training labels, trained model, and fold index.
                X_train (pandas.DataFrame): Training data for the current fold.
                y_train (pandas.Series): Training labels for the current fold.
                ensemble_model (sklearn.base.BaseEstimator): Trained ensemble model for the current fold.
                fold_idx (int): Index of the current fold.
            """

            for X_train, y_train, ensemble_model, fold_idx in learning_curve_data:
                plot_learning_curve(
                    estimator=ensemble_model,
                    X=X_train,
                    y=y_train,
                    title=f"Learning Curve for Fold {fold_idx}",
                    train_sizes=np.linspace(0.1, 1.0, 6),
                    cv=6
                )
        
        
        
        logging.info(f"Training and evaluation completed for all folds.\n")
        # Calculate and log the overall test accuracy
        mean_test_accuracy = np.mean(fold_test_accuracies)
        logging.info(f"Overall Test Accuracy: {mean_test_accuracy * 100:.2f}%")
                
      

    except Exception as e:
        logging.error(f"An error occurred: {e}")



# Call the main function
if __name__ == "__main__":
    main()
