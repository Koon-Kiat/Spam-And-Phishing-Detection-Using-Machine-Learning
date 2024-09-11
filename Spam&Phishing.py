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
from sklearn.decomposition import PCA  # Principal Component Analysis

# Ensemble classifiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Text feature extraction
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import (  # Model evaluation
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # Model selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample  # Resampling utilities

# Spell checking
from spellchecker import SpellChecker  # Spell checking
from torch.utils.data import DataLoader, Dataset  # Data handling in PyTorch

# Progress bar
from tqdm import tqdm  # Progress bar for loops

# Transformers library
# BERT models and training utilities
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from wordcloud import WordCloud  # Generate word clouds

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
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ',
    level=logging.INFO
)
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def drop_unnamed_column(df, dataset_name):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        logging.info(f"Dropped 'Unnamed: 0' column from {dataset_name}.")
    return df


def check_and_remove_missing_values(df, dataset_name):
    check_missing_values = df.isnull().sum()
    total_missing_values = check_missing_values.sum()
    logging.info(f"Total missing values: {total_missing_values}")

    logging.info(f"Removing missing values from {dataset_name}...")
    df = df.dropna()
    logging.info(f"Total number of rows after removing missing values from {
                 dataset_name}: {df.shape[0]}")

    return df


def remove_duplicates(df, column_name):
    logging.info(f"Removing duplicate data....")
    num_duplicates_before = df.duplicated(
        subset=[column_name], keep=False).sum()
    df_cleaned = df.drop_duplicates(subset=[column_name], keep='first')
    num_duplicates_after = df_cleaned.duplicated(
        subset=[column_name], keep=False).sum()
    duplicates_removed = num_duplicates_before - num_duplicates_after

    logging.info(f"Total number of rows identified as duplicates based on '{
                 column_name}': {num_duplicates_before}")
    logging.info(f"Number of rows removed due to duplication: {
                 duplicates_removed}")

    return df_cleaned


def process_dataset(df, column_name, dataset_name):
    logging.info(f"Total number of rows in {
                 dataset_name} DataFrame: {df.shape[0]}")

    # Drop the 'Unnamed: 0' column
    df = drop_unnamed_column(df, dataset_name)

    # Check for missing and duplicate values
    df = check_and_remove_missing_values(df, dataset_name)

    # Check for duplicate values and remove them
    df_cleaned = remove_duplicates(df, column_name)
    logging.info(f"Total number of rows remaining in the {
                 dataset_name}: {df_cleaned.shape[0]}\n")
    logging.debug(f"{dataset_name} after removing duplicates:\n{
                  df_cleaned.head()}\n")

    return df_cleaned


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

    # Filter out labels with 0% for original data
    original_data = [(original_safe_count / original_total_count, 'Safe Emails', original_safe_count),
                     (original_phishing_count / original_total_count,
                      'Phishing Emails', original_phishing_count),
                     (original_spam_count / original_total_count, 'Spam Emails', original_spam_count)]
    original_data = [item for item in original_data if item[0] > 0]

    # Filter out labels with 0% for cleaned data
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
        wedges, texts, autotexts = axs[1].pie(cleaned_sizes, labels=cleaned_labels, autopct='%.0f%%', colors=[
                                              'blue', 'red', 'green'], startangle=140, textprops={'fontsize': 14, 'color': 'black'})
        axs[1].set_title(
            'Distribution of Safe, Phishing, and Spam Emails (After Removing Duplicates)', color='black')

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
    ax.set_title(
        'Safe vs Phishing vs Spam Email Count (Original vs Remove Duplicates)', color='black')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Safe', 'Phishing', 'Spam'], color='black')
    ax.legend()

    for p in bar1 + bar2:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black')

    plt.show()


class EmailHeaderExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")

    def clean_links(self, links: List[str]) -> List[str]:
        cleaned_links = []
        for link in links:
            # Remove single quotes and brackets, then clean new lines and extra spaces
            # Remove single quotes, brackets, and whitespace
            link = re.sub(r'[\'\[\]\s]+', '', link)
            # Replace \n and repeating new lines with a single space
            link = re.sub(r'\\n+', ' ', link)
            link = link.strip()  # Trim leading and trailing spaces
            if link:  # Avoid appending empty links
                cleaned_links.append(link)
        return cleaned_links

    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        # Regex to capture full email addresses in the format Name <email@domain.com>
        from_match = re.search(
            r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(
            r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(
            r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)

        from_header = from_match.group(1) if from_match else None
        to_header = to_match.group(1) if to_match else None
        mail_to_header = mail_to_match.group(1) if mail_to_match else None

        return {'From': from_header, 'To': to_header, 'Mail-To': mail_to_header}

    def extract_body_content(self, email_message: EmailMessage) -> str:
        # Extract body content from different parts of the email
        body_content = ""
        if email_message.is_multipart():
            for part in email_message.iter_parts():
                if part.get_content_type() == 'text/plain':
                    body_content += part.get_payload(
                        decode=True).decode(errors='ignore')
                elif part.get_content_type() == 'text/html':
                    body_content += part.get_payload(
                        decode=True).decode(errors='ignore')
        else:
            body_content = email_message.get_payload(
                decode=True).decode(errors='ignore')
        return body_content

    def extract_headers(self) -> pd.DataFrame:
        headers_list: List[Dict[str, Union[str, List[str]]]] = []

        # Add a progress bar for email processing
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                # Parse the email
                email_message = BytesParser(policy=policy.default).parsebytes(
                    email_text.encode('utf-8'))

                # Extract 'From', 'To', and 'Mail-To' headers
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get(
                    'Mail-To') if email_message.get('Mail-To') else None

                # Fallback to inline header extraction if headers are not present
                if not from_header or not to_header:
                    inline_headers = self.extract_inline_headers(email_text)
                    from_header = inline_headers['From'] or from_header
                    to_header = inline_headers['To'] or to_header
                    mail_to_header = inline_headers['Mail-To'] or mail_to_header

                # Clean 'From', 'To', and 'Mail-To' headers
                from_header = from_header if from_header else None
                to_header = to_header if to_header else None
                mail_to_header = mail_to_header if mail_to_header else None

                # Extract body content
                body_content = self.extract_body_content(email_message)
                logging.debug(f"Email body content: {body_content}")

                # Extract links from the body content
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
        # Clean 'Text URLs' column after extraction
        self.headers_df['texturls'] = self.headers_df['texturls'].apply(
            self.clean_links)
        return self.headers_df

    def save_to_csv(self, file_path: str):
        if not self.headers_df.empty:
            # Apply the cleaning function to your DataFrame
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError(
                "No header information extracted. Please run extract_headers() first.")


def data_cleaning_and_save_text(dataset_name, df_processed, text_column, clean_file):
    logging.info(f"Text processing {dataset_name} dataset...")
    processor = TextProcessor()
    df_clean = processor.transform(
        df_processed[text_column], df_processed['label'])
    processor.save_to_csv_cleaned(df_clean, clean_file)
    logging.info("Text processing and saving completed.")
    logging.info(f"DataFrame columns after data cleaning: {
                 df_clean.columns}\n")
    return df_clean

# Data cleaning


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
        # Define additional punctuation or symbols to remove
        extra_punctuation = '“”‘’—–•·’'

        # Combine standard punctuation and extra punctuation
        all_punctuation = string.punctuation + extra_punctuation

        # Remove all punctuation from the text
        return text.translate(str.maketrans('', '', all_punctuation))

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, words_list):
        return [w for w in words_list if w.lower() not in self.stop_words]

    def lemmatize(self, words_list):
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    def remove_urls(self, text):
        # Updated to remove non-standard URL formats (e.g., 'http ww newsisfree com')
        return re.sub(r'(http[s]?|ftp):\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    def remove_custom_urls(self, text):
        # Catch patterns like 'http ww' or 'www.' that are incomplete
        return re.sub(r'\b(?:http|www)[^\s]*\b', '', text)

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
        # Remove common email headers
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:', 'Reply-To:', 'Content-Type:', 'Return-Path:', 'Message-ID:',
                   'Received:', 'MIME-Version:', 'Delivered-To:', 'Authentication-Results:', 'DKIM-Signature:', 'X-', 'Mail-To:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)
        return text

    def remove_emails(self, text):
        # Regex pattern to match emails with or without spaces around "@"
        email_pattern_with_spaces = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # Regex pattern to match emails without spaces
        email_pattern_no_spaces = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        # Combine both patterns using the OR operator
        combined_pattern = f"({email_pattern_with_spaces}|{
            email_pattern_no_spaces})"

        # Perform the substitution
        return re.sub(combined_pattern, '', text)

    def remove_time(self, text):
        # Regex to match various time patterns
        time_pattern = r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?: ?[APMapm]{2})?(?: [A-Z]{1,5})?\b'
        return re.sub(time_pattern, '', text)

    def remove_months(self, text):
        # List of full and shortened month names
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        # Regex to match months
        months_regex = r'\b(?:' + '|'.join(months) + r')\b'
        return re.sub(months_regex, '', text, flags=re.IGNORECASE)

    def remove_dates(self, text):
        # Regex to match various date formats
        date_pattern = (
            # Example: Mon, 2 Sep 2002
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|'
            # Example: 20-09-2002, Sep 13 2002
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|'
            r'\b(?:\d{1,2}\s[A-Za-z]+\s\d{4})\b|'  # Example: 01 September 2002
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})\b'  # Example: 24/08/2002
        )
        return re.sub(date_pattern, '', text, flags=re.IGNORECASE)

    def remove_timezones(self, text):
        # Regex to match time zones (e.g., PST, EST, GMT, UTC)
        timezone_pattern = r'\b(?:[A-Z]{2,4}[+-]\d{2,4}|UTC|GMT|PST|EST|CST|MST)\b'
        return re.sub(timezone_pattern, '', text)

    def remove_multiple_newlines(self, text):
        # Replace multiple newlines with a single newline
        return re.sub(r'\n{2,}', '\n', text)

    def remove_words(self, text):
        # Combine both words using the | (OR) operator in regex
        return re.sub(r'\b(url|original message)\b', '', text, flags=re.IGNORECASE)

    def remove_single_characters(self, text):
        # Remove single characters that are not part of a word
        return re.sub(r'\b\w\b', '', text)

    def remove_repetitive_patterns(self, text):
        # Combine patterns for 'nt+ts?', repetitive 'n' or 'nt', and 't+', 'n+', 'nt+'
        return re.sub(r'\b(nt+ts?|n+|t+|nt+)\b', '', text)

    def lowercase_text(self, text):
        return text.lower()

    def remove_bullet_points_and_symbols(self, text):
        # List of bullet points and similar symbols
        symbols = ['•', '◦', '◉', '▪', '▫', '●', '□', '■',
                   '✦', '✧', '✪', '✫', '✬', '✭', '✮', '✯', '✰']

        # Remove all occurrences of these symbols
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
            logging.info(f"Total amount of text processed: {
                         len(cleaned_text_list)}")
            return pd.DataFrame({'cleaned_text': cleaned_text_list, 'label': y})
        else:
            logging.info(f"Total amount of text processed: {
                         len(cleaned_text_list)}")
            return pd.DataFrame({'cleaned_text': cleaned_text_list})

    def save_to_csv_cleaned(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")

# Plot word clouds


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
        wordcloud = WordCloud(width=width, height=height, background_color=background_color,
                              max_words=max_words, stopwords=stopwords, colormap=colormap).generate(unique_string)

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


def extract_bert_features(dataset_name, df_clean):
    logging.info(f"BERT feature extraction from {dataset_name} dataset...")
    feature_extractor = BERTFeatureExtractor()
    texts = df_clean['cleaned_text'].tolist()
    bert_features = feature_extractor.extract_features(texts)
    logging.info(f"BERT feature extraction from {dataset_name} completed.\n")
    return bert_features


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
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Ensure model is on the right device

    def extract_features(self, texts, batch_size=16):
        features = []
        self.model.eval()

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT features", leave=True):
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]
                tokens = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_features = outputs.last_hidden_state.mean(
                    dim=1).cpu().numpy()  # Move back to CPU
                features.extend(batch_features)

        # Convert features to DataFrame
        return features

# Split the data into training and testing sets


def split_data(features_df, test_size=0.2, random_state=42):
    logging.info("Splitting the data into training and testing sets...")
    # Assuming 'label' is the column name for labels in features_df
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Handle data imbalance


def handle_data_imbalance(X_train, y_train, random_state=42):
    logging.info("Handling data imbalance...")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced

# Train and evaluate the ensemble model


def train_and_evaluate_ensemble(X_train_balanced, y_train_balanced, X_test, y_test):
    # Initialize the classifiers
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    logreg_model = LogisticRegression(class_weight='balanced', random_state=42)

    # Hyperparameter tuning for RandomForest
    param_grid = {
        'n_estimators': [100],  # Reduced number of estimators
        'max_depth': [None],    # Single value for max_depth
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }

    def profile_grid_search():
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        return grid_search

    # Timing the GridSearchCV
    start_time = time.time()
    best_rf_model = profile_grid_search().best_estimator_
    end_time = time.time()
    logging.info(f"GridSearchCV took {end_time - start_time:.2f} seconds\n")

    # Initialize VotingClassifier (Ensemble)
    ensemble_model = VotingClassifier(estimators=[
        ('rf', best_rf_model),
        ('logreg', logreg_model)
    ], voting='soft')

    # Train the ensemble model with progress bar
    for _ in tqdm(range(1), desc="Training ensemble model"):
        ensemble_model.fit(X_train_balanced, y_train_balanced)

    # Make predictions
    y_train_pred = ensemble_model.predict(
        X_train_balanced)  # Predictions on the training set
    y_test_pred = ensemble_model.predict(
        X_test)    # Predictions on the test set

    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Safe', 'Phishing', 'Spam']

    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    # Print classification report for training data
    print("Classification Report for Training Data:")
    print(classification_report(y_train_balanced,
          y_train_pred, target_names=target_names))

    # Print classification report for test data
    print("\nClassification Report for Test Data:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))


def log_label_percentages(df, dataset_name):
    total_count = len(df)
    label_counts = df['label'].value_counts(normalize=True) * 100
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Total count: {total_count}")

    # Sort label counts by label value
    sorted_label_counts = label_counts.sort_index()

    # Get the number of unique labels
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


# Main processing function
def main():
    # Use relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'CEAS_08.csv')
    extracted_email_file = os.path.join(
        base_dir, 'Extracted Data', 'SpamAssassinExtractedEmailHeader.csv')
    clean_ceas_file = os.path.join(
        base_dir, 'Extracted Data', 'CleanedCEAS_08Text.csv')
    clean_spamassassin_file = os.path.join(
        base_dir, 'Extracted Data', 'CleanedSpamAssassinText.csv')
    MergedHeaderDataset = os.path.join(
        base_dir, 'Extracted Data', 'MergedHeaderDataset.csv')

    df_ceas = pd.read_csv(dataset)
    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df_spamassassin = dataset.to_pandas()

    try:
        '''Changing the labels to match the labeling scheme'''
        # Change labels to match the labeling scheme
        df_spamassassin['label'] = df_spamassassin['label'].map({1: 0, 0: 2})

        # Calculate the value counts
        label_counts = df_spamassassin['label'].value_counts()

        # Compute the percentages
        total_emails = label_counts.sum()
        spam_percentage = (label_counts[2] / total_emails) * 100
        safe_percentage = (label_counts[0] / total_emails) * 100

        # Print the percentages
        logging.info(f"Percentage of Spam emails: {spam_percentage:.2f}%")
        logging.info(f"Percentage of Safe emails: {safe_percentage:.2f}%\n")

        '''Removing duplicates and missing values from the datasets'''
        df_processed_ceas = process_dataset(df_ceas, 'body', 'CEAS_08')
        df_processed_spamassassin = process_dataset(
            df_spamassassin, 'text', 'SpamAssassin')
        log_label_percentages(df_processed_ceas, 'CEAS_08')
        log_label_percentages(df_processed_spamassassin, 'SpamAssassin')
        combined__percentage_df = pd.concat(
            [df_processed_ceas, df_processed_spamassassin])
        log_label_percentages(combined__percentage_df,
                              'Combined CEAS_08 and SpamAssassin')

        '''Visualizing data before and after removing duplicates (SpamAssassin)'''
        # visualize_data(df_spamassassin, df_processed_spamassassin)

        '''Extracting email headers from the SpamAssassin dataset after removing duplicates'''
        # Extract email header information from the cleaned spamassassin dataset
        header_extractor = EmailHeaderExtractor(df_processed_spamassassin)
        spamassassin_headers_df = header_extractor.extract_headers()
        header_extractor.save_to_csv(extracted_email_file)
        logging.info(
            "Email header extraction and saving from Spam Assassin completed.\n")

        '''Data cleaning and saving the cleaned text data'''
        df_clean_ceas = data_cleaning_and_save_text(
            "CEAS_08", df_processed_ceas, 'body', clean_ceas_file)
        df_clean_spamassassin = data_cleaning_and_save_text(
            "Spam Assassin", df_processed_spamassassin, 'text', clean_spamassassin_file)

        '''Visualizing data after cleaning the text data'''
        # logging.info("Plotting word clouds for the cleaned datasets...")
        # plot_word_cloud(df_remove_duplicate['text'], "Original Dataset")
        # plot_word_cloud(df_clean_ceas['cleaned_text'], "Cleaned CEAS_08 Dataset")
        # plot_word_cloud(df_clean_spamassassin['cleaned_text'], "Cleaned Spam Assassin Dataset")
        # logging.info("Word clouds plotted successfully.\n")

        '''Feature extraction using BERT'''
        ceas_bert_features = extract_bert_features("CEAS_08", df_clean_ceas)
        spamassassin_bert_features = extract_bert_features(
            "Spam Assassin", df_clean_spamassassin)

        '''Merging the datasets'''
        # Apply the function to create the 'urls' column
        spamassassin_headers_df['urls'] = spamassassin_headers_df['texturls'].apply(
            count_urls)

        # Check for length match between df_processed_spamassassin and spamassassin_headers_df
        if len(df_processed_spamassassin) != len(spamassassin_headers_df):
            raise ValueError(
                "The lengths of df_processed_spamassassin and spamassassin_headers_df do not match.")

        # Merge df_processed_spamassassin with spamassassin_headers_df
        merged_spamassassin = pd.concat([spamassassin_headers_df.reset_index(
            drop=True), df_processed_spamassassin[['label']].reset_index(drop=True)], axis=1)

        # Ensure the total amount of data remains the same
        assert len(merged_spamassassin) == len(
            df_processed_spamassassin), "Data loss detected in merging spamassassin data."

        # Align the columns of df_processed_ceas to match the final desired structure
        # Add mailto column with None values
        df_processed_ceas['mailto'] = None
        # Add texturls column with None values
        df_processed_ceas['texturls'] = None

        # Ensure the label column in df_processed_ceas is correctly populated
        # Assuming df_processed_ceas already has a 'label' column with correct values
        if 'label' not in df_processed_ceas.columns:
            # Add label column with None values if it doesn't exist
            df_processed_ceas['label'] = None

        # Concatenate the DataFrames
        final_df = pd.concat(
            [merged_spamassassin, df_processed_ceas], ignore_index=True)

        # Ensure the total amount of data remains the same
        assert len(final_df) == len(df_processed_spamassassin) + \
            len(df_processed_ceas), "Data loss detected in final merging."

        # Drop the 'date' column if it exists
        if 'date' in final_df.columns:
            final_df = final_df.drop(columns=['date'])

        # Drop the 'texturls' column since we now have 'urls'
        if 'texturls' in final_df.columns:
            final_df = final_df.drop(columns=['texturls'])

        # Reorder columns to match the desired structure, excluding 'date' and 'texturls'
        final_df = final_df[['sender', 'receiver',
                             'mailto', 'subject', 'urls', 'label']]

        # Calculate and log the percentage of each label
        log_label_percentages(final_df, "Final Merged Dataset")

        # Set pandas display options
        pd.set_option('display.max_columns', None)  # Show all columns
        # Adjust width to fit content
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 100)  # Limit column width

        # Save the final DataFrame to a CSV file
        final_df.to_csv(MergedHeaderDataset, index=False)

        '''Verify the label columns have merge'''
        combined_labels = combined__percentage_df['label'].reset_index(
            drop=True)
        final_labels = final_df['label'].reset_index(drop=True)

        labels_match = combined_labels.equals(final_labels)

        if labels_match:
            print(
                "The label columns match between the combined percentage DataFrame and final DataFrame.")
            # Log label percentages for combined data
            log_label_percentages(combined__percentage_df,
                                  "Combined Percentage DataFrame")
            # Log label percentages for final data
            log_label_percentages(final_df, "Final DataFrame")
        else:
            print(
                "The label columns do not match between the combined percentage DataFrame and final DataFrame.")
            # Optionally, you can also inspect mismatches
            mismatched_labels = combined_labels[combined_labels != final_labels]
            print(f"Mismatched labels:\n{mismatched_labels}")

            # Log label percentages for both DataFrames to inspect discrepancies
            log_label_percentages(combined__percentage_df,
                                  "Combined Percentage DataFrame")
            log_label_percentages(final_df, "Final DataFrame")

        logging.info("Verification complete.\n")

        '''Preprocessing the final merged dataset'''
        # Fill missing values for specific columns where necessary
        final_df['sender'] = final_df['sender'].fillna('unknown')
        final_df['receiver'] = final_df['receiver'].fillna('unknown')
        final_df['mailto'] = final_df['mailto'].fillna('unknown')
        final_df['subject'] = final_df['subject'].fillna('unknown')

        # Fill missing values in 'urls' with 0
        final_df['urls'] = final_df['urls'].fillna(0)

        # Check for missing values in all columns
        missing_values = final_df.isnull().sum()
        logging.info(f"Missing values in each column:\n{missing_values}")

        # Define the columns for preprocessing
        categorical_columns = ['sender', 'receiver', 'mailto', 'subject']
        numerical_columns = ['urls']

        # Define the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                # Categorical columns: Use OneHotEncoder
                ('cat', OneHotEncoder(handle_unknown='ignore',
                 max_categories=100), categorical_columns),

                # Numerical columns: Use StandardScaler
                ('num', StandardScaler(), numerical_columns)
            ]
        )

        # Apply the preprocessing pipeline
        preprocessed_data = preprocessor.fit_transform(final_df)

        # Convert preprocessed data to a DataFrame and ensure column names are strings
        preprocessed_df = pd.DataFrame(
            preprocessed_data.toarray(), columns=preprocessor.get_feature_names_out())
        preprocessed_df.columns = preprocessed_df.columns.astype(
            str)  # Ensure all column names are strings

        # Add the label column back to the preprocessed DataFrame
        preprocessed_df['label'] = final_df['label'].values

        # Display some information about the preprocessed DataFrame
        logging.info(f"Shape of preprocessed DataFrame: {
                     preprocessed_df.shape}")
        logging.info(f"Missing values after preprocessing:\n{
                     preprocessed_df.isnull().sum()}")

        # Convert BERT features to DataFrames if needed and ensure column names are strings
        ceas_bert_features_df = pd.DataFrame(ceas_bert_features)
        ceas_bert_features_df.columns = ceas_bert_features_df.columns.astype(
            str)

        spamassassin_bert_features_df = pd.DataFrame(
            spamassassin_bert_features)
        spamassassin_bert_features_df.columns = spamassassin_bert_features_df.columns.astype(
            str)

        # Check shapes
        logging.info(f"Checking shapes:")
        logging.info(f"Shape of preprocessed_df: {preprocessed_df.shape}")
        logging.info(f"Shape of ceas_bert_features_df: {
                     ceas_bert_features_df.shape}")
        logging.info(f"Shape of spamassassin_bert_features_df: {
                     spamassassin_bert_features_df.shape}\n")

        # Verify column names
        logging.info(f"Checking column names:")
        logging.info(f"Columns in preprocessed_df: {
                     preprocessed_df.columns.tolist()}")
        logging.info(f"Columns in ceas_bert_features_df: {
                     ceas_bert_features_df.columns.tolist()}")
        logging.info(f"Columns in spamassassin_bert_features_df: {
                     spamassassin_bert_features_df.columns.tolist()}\n")

        # Check for missing values
        logging.info(f"Checking missing values:")
        logging.info(f"Missing values in preprocessed_df:\n{
                     preprocessed_df.isnull().sum()}")
        logging.info(f"Missing values in ceas_bert_features_df:\n{
                     ceas_bert_features_df.isnull().sum()}")
        logging.info(f"Missing values in spamassassin_bert_features_df:\n{
                     spamassassin_bert_features_df.isnull().sum()}\n")

        # Check the final features DataFrame
        final_features = pd.concat(
            [preprocessed_df, spamassassin_bert_features_df, ceas_bert_features_df], axis=1)

        # Check for missing values in final_features
        logging.info(f"Missing values in final_features:\n{
                     final_features.isnull().sum()}")

        # Check for duplicates
        logging.info(f"Number of duplicate rows in final_features: {
                     final_features.duplicated().sum()}")

        # Cross-check counts
        logging.info(f"Number of samples in final_features: {
                     final_features.shape[0]}")
        logging.info(f"Number of samples in preprocessed_df: {
                     preprocessed_df.shape[0]}")

        # Convert all column names in final_features to strings
        final_features.columns = final_features.columns.astype(str)

        # Ensure no misalignment between preprocessed_df and BERT feature dataframes
        if final_features.shape[0] != final_df.shape[0]:
            logging.info(
                "Error: Row count mismatch between preprocessed data and BERT features.")

        # Handle missing values in the final merged dataset
        if final_features.isnull().sum().sum() > 0:
            logging.info("Warning: Missing values found after processing!\n")

        # Impute missing values
        # Use 'mean' for numerical data
        imputer = SimpleImputer(strategy='mean')
        final_features = pd.DataFrame(imputer.fit_transform(
            final_features), columns=final_features.columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(final_features)
        logging.info(f"Data split into training and testing sets.\n")

        # Handle data imbalance
        X_train_balanced, y_train_balanced = handle_data_imbalance(
            X_train, y_train)
        logging.info(f"Data imbalance handled.\n")

        # Train and evaluate the ensemble model
        train_and_evaluate_ensemble(
            X_train_balanced, y_train_balanced, X_test, y_test)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Call the main function
if __name__ == "__main__":
    main()
