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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
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
            body_content = email_message.get_payload(
                decode=True).decode(errors='ignore')
        return body_content



    def extract_headers(self) -> pd.DataFrame:
        headers_list: List[Dict[str, Union[str, List[str]]]] = []
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                email_message = BytesParser(policy=policy.default).parsebytes(email_text.encode('utf-8'))
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get(
                    'Mail-To') if email_message.get('Mail-To') else None



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
                url_pattern = r'https?:\/\/[^\s\'"()<>]+'
                links = re.findall(url_pattern, body_content)
                links = self.clean_links(links)



                headers_list.append({
                    'sender': from_header,
                    'receiver': to_header,
                    'mailto': mail_to_header,
                    'texturls': links
                })
            except Exception as e:
                logging.error(f"Error parsing email: {e}")
                headers_list.append(
                    {'sender': None, 'receiver': None, 'mailto': None, 'texturls': []})
        self.headers_df = pd.DataFrame(headers_list)
        self.headers_df['texturls'] = self.headers_df['texturls'].apply(
            self.clean_links)
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
    


    def save_features(self, features, features_path):
        logging.info(f"Saving features to {features_path}.")
        np.save(features_path, features)
        logging.info(f"Features saved to {features_path}.")



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
        wedges, texts, autotexts = axs[0].pie(original_sizes, labels=original_labels, autopct='%.0f%%', colors=[
                                              'blue', 'red', 'green'], startangle=140, textprops={'fontsize': 14, 'color': 'black'})
        axs[0].set_title(
            'Distribution of Safe, Phishing, and Spam Emails (Original)', color='black')

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
        logging.info(f"File {file_path} already exists. Loading from file.\n")
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
        logging.info(f"Data cleaning and saving to {file_path} completed.\n")
        return cleaned_df
   


def load_or_extract_headers(df, file_path, extractor_class):
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading from file.")
        return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist. Extracting headers.")
        header_extractor = extractor_class(df)
        headers_df = header_extractor.extract_headers()
        header_extractor.save_to_csv(file_path)
        logging.info(f"Email header extraction and saving to {file_path} completed.\n")
        return headers_df



def load_or_extract_bert_features(dataset_name, df_clean, features_path):
    logging.info(f"BERT feature extraction from {dataset_name} dataset...")
    feature_extractor = BERTFeatureExtractor()
    saved_features = feature_extractor.load_features(features_path)
    if saved_features is not None:
        logging.info(f"Loaded saved BERT features from {features_path}.\n")
        return saved_features
    feature_extractor = BERTFeatureExtractor()
    texts = df_clean['cleaned_text'].tolist()
    bert_features = feature_extractor.extract_features(texts)
    feature_extractor.save_features(bert_features, features_path)
    logging.info(f"BERT feature extraction from {dataset_name} completed.\n")
    return bert_features



def split(features_df, test_size=0.2, random_state=42):
    logging.info("Splitting the data into training and testing sets...")
    X = features_df.drop(columns=['label'])
    y = features_df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Ensure the class distribution is preserved
    )
    return X_train, X_test, y_train, y_test



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
    logging.info("Handling data imbalance...")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced



def model_training(X_train_balanced, y_train_balanced, X_test, y_test):
    logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    gbm_model = GradientBoostingClassifier(random_state=42)

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
        rf_model.fit(X_train_balanced, y_train_balanced)
        y_train_pred = rf_model.predict(X_train_balanced)
        return accuracy_score(y_train_balanced, y_train_pred)


    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=5)


    # Retrieve the best model after optimization
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")


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
    best_rf_model.fit(X_train_balanced, y_train_balanced)


    # Define the ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ('rf', best_rf_model),
        ('logreg', logreg_model)
        #('gbm', gbm_model)
    ], voting='soft')


    # Train the ensemble model with a progress bar
    for _ in tqdm(range(1), desc="Training ensemble model"):
        ensemble_model.fit(X_train_balanced, y_train_balanced)


    # Make predictions
    y_train_pred = ensemble_model.predict(X_train_balanced)
    y_test_pred = ensemble_model.predict(X_test)


    # Evaluate the model
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Safe', 'Phishing', 'Spam']


    # Logging the performance metrics
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_test_pred)))
    print("Classification Report for Training Data:\n" + classification_report(y_train_balanced, y_train_pred, target_names=target_names))
    print("\nClassification Report for Test Data:\n" + classification_report(y_test, y_test_pred, target_names=target_names))



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
        if not rows_with_missing.empty:
            logging.info(f"Examples of rows with missing values in {df_name}:")
            for idx, row in rows_with_missing.head(num_rows).iterrows():
                logging.info(f"Row {idx}:{row}")
        else:
            logging.info(f"No rows with missing values found in {df_name} after initial check.")


# Main processing function
def main():
    # Use relative paths to access the datasets and save the extracted data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'CEAS_08.csv')
    ExtractedSpamAssassinEmailHeaderFile = os.path.join(base_dir, 'Extracted Data', 'SpamAssassinExtractedEmailHeader.csv')
    CleanedCeasFile = os.path.join(base_dir, 'Extracted Data', 'CleanedCEAS_08Text.csv')
    CleanedSpamAssassinFile = os.path.join(base_dir, 'Extracted Data', 'CleanedSpamAssassinText.csv')
    MergedHeaderDataset = os.path.join(base_dir, 'Extracted Data', 'MergedHeaderDataset.csv')
    CEASExtractedFeatures = os.path.join(base_dir, 'Extracted Features', 'CEASExtractedBertFeatures.npy')
    SpamAssassinExtractedFeatures = os.path.join(base_dir, 'Extracted Features', 'SpamAssassinExtractedBertFeatures.npy')


    # Load the datasets
    df_ceas = pd.read_csv(dataset)
    dataset = load_dataset('talby/spamassassin',split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()


    try:
        # Change labels to match the labeling scheme
        df_spamassassin['label'] = df_spamassassin['label'].map({1: 0, 0: 2})


        # Removing duplicates and missing values
        processor_spamassassin = DatasetProcessor(df_spamassassin, 'text', 'SpamAssassin')
        df_processed_spamassassin = processor_spamassassin.process_dataset()
        processor_ceas = DatasetProcessor(df_ceas, 'body', 'CEAS_08')
        df_processed_ceas = processor_ceas.process_dataset()
        log_label_percentages(df_processed_ceas, 'CEAS_08')
        log_label_percentages(df_processed_spamassassin, 'SpamAssassin')
        combined_percentage_df = pd.concat([df_processed_spamassassin, df_processed_ceas], axis=0, ignore_index=True) # Combine the two DataFrames: SpamAssassin + CEAS_08
        log_label_percentages(combined_percentage_df,'Combined CEAS_08 and SpamAssassin')


        # Visualizing before and after removing duplicates
        # visualize_data(df_spamassassin, df_processed_spamassassin)


        # Extracting email headers from the SpamAssassin dataset after removing duplicates
        spamassassin_headers_df = load_or_extract_headers(df_processed_spamassassin, ExtractedSpamAssassinEmailHeaderFile, EmailHeaderExtractor)
        logging.info("Email header extraction and saving from Spam Assassin completed.\n")


        # Data cleaning and saving the cleaned text data
        df_clean_ceas = load_or_clean_data("CEAS_08", df_processed_ceas, 'body', CleanedCeasFile, data_cleaning)
        df_clean_spamassassin = load_or_clean_data("Spam Assassin", df_processed_spamassassin, 'text', CleanedSpamAssassinFile, data_cleaning)


        # Visualizing data after cleaning the text data
        # logging.info("Plotting word clouds for the cleaned datasets...")
        # plot_word_cloud(df_remove_duplicate['text'], "Original Dataset")
        # plot_word_cloud(df_clean_ceas['cleaned_text'], "Cleaned CEAS_08 Dataset")
        # plot_word_cloud(df_clean_spamassassin['cleaned_text'], "Cleaned Spam Assassin Dataset")
        # logging.info("Word clouds plotted successfully.\n")
        

        # Extract or load BERT features
        ceas_bert_features = load_or_extract_bert_features("CEAS_08", df_clean_ceas, CEASExtractedFeatures)
        spamassassin_bert_features = load_or_extract_bert_features("Spam Assassin", df_clean_spamassassin, SpamAssassinExtractedFeatures)


        # Merging the datasets between SpamAssassin
        spamassassin_headers_df['urls'] = spamassassin_headers_df['texturls'].apply(count_urls) # Apply the function to create the 'urls' column
        if len(df_processed_spamassassin) != len(spamassassin_headers_df):
            raise ValueError("The lengths of df_processed_spamassassin and spamassassin_headers_df do not match.") # Check for length match between df_processed_spamassassin and spamassassin_headers_df
        '''
        spamassassin_headers_df columns: ['sender', 'receiver', 'mailto', 'texturls', 'urls']
        df_processed_spamassassin columns: ['label', 'group', 'text']
        '''
        merged_spamassassin = pd.concat([spamassassin_headers_df.reset_index(drop=True), df_processed_spamassassin[['label']].reset_index(drop=True)], axis=1) # Add the 'label' column from df_processed_spamassassin
        assert len(merged_spamassassin) == len(df_processed_spamassassin), "Data loss detected in merging spamassassin data." # Ensure the total amount of data remains the same
        log_label_percentages(merged_spamassassin, "Merged SpamAssassin Dataset")

        # Merging the datasets between SpamAssassin and CEAS_08
        df_processed_ceas['mailto'] = None # Add mailto column with None values
        df_processed_ceas['texturls'] = None  # Add texturls column with None values
        logging.info(f"Columns in merged_spamassassin: {merged_spamassassin.columns.tolist()}")
        '''
        merged_spamassassin columns: ['sender', 'receiver', 'mailto', 'texturls', 'urls', 'label']
        df_processed_ceas columns: ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
        '''
        final_df = pd.concat([merged_spamassassin, df_processed_ceas], axis=0, ignore_index=True) # Concatenate the DataFrames


        # Verifcation of the final merged dataset
        assert len(final_df) == len(df_processed_spamassassin) + \
            len(df_processed_ceas), "Data loss detected in final merging."  # Ensure the total amount of data remains the same
        if 'date' in final_df.columns:
            final_df = final_df.drop(columns=['date']) # Drop the 'date' column if it exists
        if 'texturls' in final_df.columns:
            final_df = final_df.drop(columns=['texturls']) # Drop the 'texturls' column since we now have 'urls'
        logging.info(f"Columns in Final Merged dataset: {final_df.columns.tolist()}\n")
        '''
        Columns in merged dataset: ['sender', 'receiver', 'mailto', 'urls', 'label', 'subject', 'body'] 
        '''
        final_df = final_df[['sender', 'receiver', 'mailto', 'subject', 'urls', 'label']]  # Reorder columns to match the desired structure, excluding 'date' and 'texturls'
        log_label_percentages(final_df, "Final Merged Dataset") # Calculate and log the percentage of each label
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)  # Limit column width
        final_df.to_csv(MergedHeaderDataset, index=False)


        # Verification of the final merged dataset again
        logging.info("Verifying the Final Merged Dataset...")
        combined_labels = combined_percentage_df['label'].reset_index(drop=True)
        final_labels = final_df['label'].reset_index(drop=True)
        labels_match = combined_labels.equals(final_labels)
        if labels_match:
            logging.info("The label columns match between the combined percentage DataFrame and final DataFrame.")
            log_label_percentages(combined_percentage_df, "Combined Percentage DataFrame At The Start")
            log_label_percentages(final_df, "Final Merged DataFrame")
        else:
            logging.info("The label columns do not match between the combined percentage DataFrame and final DataFrame.")
            mismatched_labels = combined_labels[combined_labels != final_labels]
            logging.info(f"Mismatched labels:\n{mismatched_labels}")
            log_label_percentages(combined_percentage_df,"Combined Percentage DataFrame At The Start")
            log_label_percentages(final_df, "Final Merged DataFrame")
        logging.info("Verification complete.\n")


        # Preprocessing the final merged dataset
        logging.info("Preprocessing the Final Merged Dataset...")
        final_df['sender'] = final_df['sender'].fillna('unknown')
        final_df['receiver'] = final_df['receiver'].fillna('unknown')
        final_df['mailto'] = final_df['mailto'].fillna('unknown')
        final_df['subject'] = final_df['subject'].fillna('unknown')
        final_df['urls'] = final_df['urls'].fillna(0) # Fill missing values in 'urls' with 0


        # Check for missing values in all columns
        check_missing_values(final_df, "Final Merged Dataframe") # No missing data
        '''
        Columns in final_df:['sender', 'receiver', 'mailto', 'subject', 'urls', 'label']
        '''
        categorical_columns = ['sender', 'receiver', 'mailto', 'subject']
        numerical_columns = ['urls']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns), # Categorical columns: Use OneHotEncoder
                ('num', StandardScaler(), numerical_columns) # Numerical columns: Use StandardScaler
            ]
        )
        preprocessed_data = preprocessor.fit_transform(final_df) # Apply the preprocessing pipeline
        preprocessed_df = pd.DataFrame(preprocessed_data, columns=preprocessor.get_feature_names_out())  # No need to call toarray()
        preprocessed_df['label'] = final_df['label'].values # Add the label column back to the preprocessed DataFrame
        num_features_preprocessed = len(preprocessor.get_feature_names_out())


        # Verification of the Preprocessed DataFrame
        logging.info(f"Number of features after preprocessing: {num_features_preprocessed}")
        logging.info(f"Number of columns in Preprocessed Dataframe after adding label: {preprocessed_df.shape[1]}")
        #logging.info(f"Columns in preprocessed_df: {preprocessed_df.columns.tolist()}")
        if preprocessed_df['label'].equals(final_df['label']):
            logging.info("The label columns in Processed Dataframe and Final Merged Dataframe match.")
        else:
            logging.error("The label columns in Processed Dataframe and Final Merged Dataframe do not match.")
        expected_feature_count = len(preprocessor.get_feature_names_out())
        actual_feature_count = preprocessed_df.shape[1] - 1
        if expected_feature_count != actual_feature_count:
            logging.error(f"Mismatch in number of features: expected {expected_feature_count}, but found {actual_feature_count}")
        expected_columns = set(preprocessor.get_feature_names_out())
        actual_columns = set(preprocessed_df.columns[:-1])
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            logging.error(f"Extra columns found in preprocessed_df: {extra_columns}")
        else:
            logging.info("No extra columns found.")
        duplicate_columns = preprocessed_df.columns[preprocessed_df.columns.duplicated()].unique()
        if len(duplicate_columns) > 0:
            logging.error(f"Duplicate column names found: {duplicate_columns}\n")
        else:
            logging.info("No duplicate column names found.\n")


        # Convert BERT features to DataFrames if needed and ensure column names are strings
        ceas_bert_features_df = pd.DataFrame(ceas_bert_features)
        ceas_bert_features_df.columns = ceas_bert_features_df.columns.astype(str)
        spamassassin_bert_features_df = pd.DataFrame(spamassassin_bert_features)
        spamassassin_bert_features_df.columns = spamassassin_bert_features_df.columns.astype(str)


        # Verification of Preprocessed Dataframe, SpamAssassin Extracted Bert Features Dataframe, and CEAS_08 Extracted Bert Features Dataframe
        logging.info(f"Checking shapes...")
        logging.info(f"Shape of Preprocessed Dataframe: {preprocessed_df.shape}")
        logging.info(f"Shape of SpamAssassin Extracted Bert Features Dataframe: {spamassassin_bert_features_df.shape}")
        logging.info(f"Shape of CEAS_08 Extracted Bert Features Dataframe: {ceas_bert_features_df.shape}\n")
        logging.info("Checking missing values:")
        check_missing_values(preprocessed_df, "Preprocessed Dataframe")
        check_missing_values(spamassassin_bert_features_df, "SpamAssassin Extracted Bert Features Dataframe")
        check_missing_values(ceas_bert_features_df, "CEAS_08 Extracted Bert Features Dataframe")
        logging.info(f"Finish checking missing values.\n")
        
        # Check the number of rows in each DataFrame
        logging.info(f"Checking number of rows in each DataFrame:")
        num_rows_preprocessed = preprocessed_df.shape[0]
        num_rows_spamassassin_bert = spamassassin_bert_features_df.shape[0]
        num_rows_ceas_bert = ceas_bert_features_df.shape[0]
        logging.info(f'Number of rows in Preprocessed Dataframe: {num_rows_preprocessed}')
        logging.info(f'Number of rows in SpamAssassin Extracted Bert Features Dataframe: {num_rows_spamassassin_bert}')
        logging.info(f'Number of rows in CEAS_08 Extracted Bert Features Dataframe: {num_rows_ceas_bert}\n')
        combined_bert_features_df = pd.concat([spamassassin_bert_features_df, ceas_bert_features_df], ignore_index=True) # Combine the BERT features DataFrames


        # Add index to the BERT features DataFrames
        preprocessed_df.reset_index(drop=True, inplace=True)
        combined_bert_features_df.reset_index(drop=True, inplace=True)
        preprocessed_df['index'] = preprocessed_df.index
        combined_bert_features_df['index'] = combined_bert_features_df.index


        # Check the number of rows in the combined BERT features DataFrame and the preprocessed DataFrame
        num_rows_combined_bert = combined_bert_features_df.shape[0]
        num_rows_preprocessed = preprocessed_df.shape[0]
        logging.info(f"Checking number of rows in the Combined BERT Features DataFrame and the Preprocessed DataFrame:")
        logging.info(f'Number of rows in Combined Bert Features of SpamAssassin and CEAS_08 Dataframe: {num_rows_combined_bert}')
        logging.info(f'Number of rows in Preprocessed Dataframe: {num_rows_preprocessed}')
        if num_rows_combined_bert == num_rows_preprocessed:
            logging.info("Both DataFrames have the same number of rows.\n")
        else:
            logging.info("Row counts do not match. Please check your DataFrames.\n")


        # Merge the preprocessed DataFrame with the combined BERT features DataFrame
        final_features = preprocessed_df.merge(combined_bert_features_df, on='index', how='left')
        final_features.drop(columns='index', inplace=True)


        # Verification of the final features DataFrame
        check_missing_values(final_features, "Final Features (All)")
        logging.info(f"Number of duplicate rows in Final Features (All): {final_features.duplicated().sum()}")
        logging.info(f"Number of samples in Final Features (All): {final_features.shape[0]}")
        logging.info(f"Number of samples in Preprocessed Dataframe: {preprocessed_df.shape[0]}")
        if final_features.shape[0] == preprocessed_df.shape[0]:
            logging.info("The number of samples in Final Features (All) and Preprocessed Dataframe match.\n")
        else:
            logging.warning("The number of samples in Final Features (All) and Preprocessed Dataframe do not match.")
        #final_features.columns = final_features.columns.astype(str) # Convert all column names in final_features to strings
        if final_features.shape[0] != final_df.shape[0]:
            logging.info("Error: Row count mismatch between Final Features (All) and Final Merged Dataframe.") # Ensure no misalignment between preprocessed_df and BERT feature dataframes
        if final_features.isnull().sum().sum() > 0:
            check_missing_values(final_features, "Final Features (All)") # Check missing values in the final merged dataset
            logging.info(f"Imputing missing values in Final Features (All)...")
            imputer = SimpleImputer(strategy='mean')
            final_features = pd.DataFrame(imputer.fit_transform(final_features), columns=final_features.columns)
            logging.info(f"Missing values imputed in Final Features (All).\n")


        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = split(final_features)
        logging.info(f"Data split into training and testing sets.\n")


        # Apply data imbalance handling and PCA
        X_train_balanced, y_train_balanced = smote(X_train, y_train)
        logging.info(f"Data imbalance handled.\n")
        #X_train_pca, y_train_pca = pca_incremental(X_train_balanced, y_train_balanced)
        #logging.info(f"PCA applied.\n")


        # Check the class distribution for the training and testing sets
        logging.info(f"Checking class distribution for the training and testing sets...")
        #logging.info(f"X_train_balanced dtypes: {X_train_balanced.dtypes}")
        #logging.info(f"y_train_balanced dtypes: {y_train_balanced.dtypes}")
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_train_balanced, y_train_balanced):
            train_distribution = y_train_balanced.iloc[train_index].value_counts().to_dict()
            test_distribution = y_train_balanced.iloc[test_index].value_counts().to_dict()
            logging.info(f"Train class distribution: {train_distribution}")
            logging.info(f"Test class distribution: {test_distribution}\n")
        logging.info("Checking for NaN values in features and labels...")
        logging.info(f"NaN values in features: {X_train.isna().sum().sum()}")
        logging.info(f"NaN values in labels: {y_train.isna().sum()}\n")
        logging.info("Checking for infinite values in features...")
        logging.info(f"Inf values in features: {np.isinf(X_train).sum().sum()}\n")



        #X_train_small = X_train_balanced[:1000]
        #y_train_small = y_train_balanced[:1000]


        # Train and evaluate the ensemble model
        model_training(X_train_balanced, y_train_balanced, X_test, y_test)



    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Call the main function
if __name__ == "__main__":
    main()
