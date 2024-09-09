# Description: This file is used to test the data cleaning and processing functions.

# Data manipulation
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis

# Data visualization
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization
from wordcloud import WordCloud  # Generate word clouds

# Operating system interfaces
import os  # Interact with the operating system

# Email parsing
import email  # Email handling
import email.policy  # Email policies
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage

# String and regular expression operations
import string  # String operations
import re  # Regular expressions

# HTML and XML parsing
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # HTML and XML parsing

# Progress bar
from tqdm import tqdm  # Progress bar for loops

# Logging
import logging  # Logging library

# Text processing
import contractions  # Expand contractions in text
import codecs  # Codec registry and base classes
import json  # JSON parsing and manipulation
import urllib.parse  # URL parsing

# Natural Language Toolkit (NLTK)
import nltk  # Natural language processing
from nltk.stem import WordNetLemmatizer  # Lemmatization
from nltk.corpus import stopwords  # Stop words
from nltk.tokenize import word_tokenize  # Tokenization

# Typing support
from typing import List, Dict, Union  # Type hints

# Concurrent execution
from concurrent.futures import ThreadPoolExecutor, as_completed  # Multithreading
from functools import lru_cache  # Least Recently Used (LRU) cache

# Spell checking
from spellchecker import SpellChecker  # Spell checking

# Machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin  # Scikit-learn base classes
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # Text feature extraction
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.model_selection import train_test_split, GridSearchCV  # Model selection
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Ensemble classifiers
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # Model evaluation
from sklearn.utils import resample  # Resampling utilities
from imblearn.over_sampling import SMOTE  # Handling imbalanced data
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Transformers library
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel, AdamW  # BERT models and training utilities

# PyTorch
import torch  # PyTorch library
from torch.utils.data import DataLoader, Dataset  # Data handling in PyTorch

# TensorFlow
import tensorflow as tf  # TensorFlow library

# Sparse matrices
from scipy.sparse import hstack, csr_matrix  # Sparse matrix operations

# Profiling and job management
import time  # Time-related functions
import cProfile  # Profiling
import joblib  # Job management

# Warnings
import warnings  # Warning control

# Datasets
from datasets import load_dataset  # Load datasets

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

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


# Remove duplicate data
def remove_duplicate(df):
    logging.info("Removing duplicate data...")
    num_duplicates_before = df.duplicated(subset=['text'], keep=False).sum()
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
    num_duplicates_after = df_cleaned.duplicated(
        subset=['text'], keep=False).sum()
    duplicates_removed = num_duplicates_before - num_duplicates_after

    logging.info(f"Total number of rows identified as duplicates based on 'text': {num_duplicates_before}")
    logging.info(f"Number of rows removed due to duplication: {duplicates_removed}")

    return df_cleaned

# Visualize data
def visualize_data_comparison(df_before, df_after):
    logging.info("Visualizing data before and after cleaning...")

    # Mapping and counting labels for before and after cleaning
    label_map = {1: 'Safe', 0: 'Phishing'}
    label_counts_before = df_before['label'].value_counts()
    label_counts_after = df_after['label'].value_counts()

    ham_count_before = label_counts_before.get(1, 0)  # Safe (ham) emails count before cleaning
    spam_count_before = label_counts_before.get(0, 0)  # Phishing (spam) emails count before cleaning
    total_before = ham_count_before + spam_count_before

    ham_count_after = label_counts_after.get(1, 0)  # Safe (ham) emails count after cleaning
    spam_count_after = label_counts_after.get(0, 0)  # Phishing (spam) emails count after cleaning
    total_after = ham_count_after + spam_count_after

    if total_before == 0 or total_after == 0:
        logging.warning("No data to visualize.")
        return

    # Plot distribution before and after cleaning
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart for data before cleaning
    data_before = [ham_count_before / total_before, spam_count_before / total_before]  # Ensure correct order: Safe, Phishing
    wedges, texts, autotexts = axes[0].pie(
        data_before, labels=['Safe', 'Phishing'], autopct='%.0f%%', colors=['blue', 'red'], startangle=140)
    axes[0].set_title('Before Cleaning')

    # Adding text labels for counts next to wedges
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        count = ham_count_before if i == 0 else spam_count_before  # Match count with wedge
        axes[0].text(x, y, f'{["Safe", "Phishing"][i]}: {count}', ha='center', va='center', fontsize=10)

    # Pie chart for data after cleaning
    data_after = [ham_count_after / total_after, spam_count_after / total_after]  # Ensure correct order: Safe, Phishing
    wedges, texts, autotexts = axes[1].pie(
        data_after, labels=['Safe', 'Phishing'], autopct='%.0f%%', colors=['blue', 'red'], startangle=140)
    axes[1].set_title('After Cleaning')

    # Adding text labels for counts next to wedges
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        count = ham_count_after if i == 0 else spam_count_after  # Match count with wedge
        axes[1].text(x, y, f'{["Safe", "Phishing"][i]}: {count}', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Bar plot for email counts before and after cleaning
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot before cleaning
    sns.countplot(x='label', data=df_before, hue='label', palette=['blue', 'red'], ax=axes[0], order=[1, 0])
    axes[0].legend_.remove()  # Remove legend since we are displaying labels directly
    axes[0].set_title('Safe vs Phishing Email Count (Before Cleaning)')
    axes[0].set_xticks([0, 1])  # Set correct tick positions
    axes[0].set_xticklabels(['Safe', 'Phishing'])
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')

    # Annotate percentages on the bars
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:  # Only annotate if there is a non-zero count
            axes[0].annotate(f'{height / total_before:.1%}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Bar plot after cleaning
    sns.countplot(x='label', data=df_after, hue='label', palette=['blue', 'red'], ax=axes[1], order=[1, 0])
    axes[1].legend_.remove()  # Remove legend since we are displaying labels directly
    axes[1].set_title('Safe vs Phishing Email Count (After Cleaning)')
    axes[1].set_xticks([0, 1])  # Set correct tick positions
    axes[1].set_xticklabels(['Safe', 'Phishing'])
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')

    # Annotate percentages on the bars
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:  # Only annotate if there is a non-zero count
            axes[1].annotate(f'{height / total_after:.1%}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.show()

# Extract email header
class EmailHeaderExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")

    def clean_links(self, links: List[str]) -> List[str]:
        cleaned_links = []
        for link in links:
            # Remove single quotes and brackets, then clean new lines and extra spaces
            link = re.sub(r'[\'\[\]\s]+', '', link)  # Remove single quotes, brackets, and whitespace
            link = re.sub(r'\\n+', ' ', link)  # Replace \n and repeating new lines with a single space
            link = link.strip()  # Trim leading and trailing spaces
            if link:  # Avoid appending empty links
                cleaned_links.append(link)
        return cleaned_links

    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        # Regex to capture full email addresses in the format Name <email@domain.com>
        from_match = re.search(r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)

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
                    body_content += part.get_payload(decode=True).decode(errors='ignore')
                elif part.get_content_type() == 'text/html':
                    body_content += part.get_payload(decode=True).decode(errors='ignore')
        else:
            body_content = email_message.get_payload(decode=True).decode(errors='ignore')
        return body_content

    def extract_headers(self) -> pd.DataFrame:
        headers_list: List[Dict[str, Union[str, List[str]]]] = []

        # Add a progress bar for email processing
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                # Parse the email
                email_message = BytesParser(policy=policy.default).parsebytes(email_text.encode('utf-8'))

                # Extract 'From', 'To', and 'Mail-To' headers
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get('Mail-To') if email_message.get('Mail-To') else None

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
                    'From': from_header,
                    'To': to_header,
                    'Mail-To': mail_to_header,
                    'Links': links
                })
            except Exception as e:
                logging.error(f"Error parsing email: {e}")
                headers_list.append({'From': None, 'To': None, 'Mail-To': None, 'Links': []})
                
        self.headers_df = pd.DataFrame(headers_list)
        # Clean 'Links' column after extraction
        self.headers_df['Links'] = self.headers_df['Links'].apply(self.clean_links)
        return self.headers_df

    def save_to_csv(self, file_path: str):
        if not self.headers_df.empty:
            # Apply the cleaning function to your DataFrame
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError("No header information extracted. Please run extract_headers() first.")

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
        return text.translate(str.maketrans('', '', string.punctuation))

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

    def remove_repeating_characters(self, text):
        return re.sub(r'(.)\1+', r'\1\1', text)

    def remove_all_html_elements(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()  
        for tag in soup.find_all(True):
            tag.attrs = {}
        return soup.get_text(separator=" ", strip=True)
    
    def remove_emails(self, text):
        # Remove email addresses
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    def remove_email_headers(self, text):
        # Remove common email headers
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:', 'Reply-To:', 'Content-Type:', 'Return-Path:', 'Message-ID:', 'Received:', 'MIME-Version:', 'Delivered-To:', 'Authentication-Results:', 'DKIM-Signature:', 'X-', 'Mail-To:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)
        return text
    
    def remove_emails(self, text):
        # Regex to match emails with or without spaces around "@"
        email_pattern = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
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
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|'  # Example: Mon, 2 Sep 2002
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|'  # Example: 20-09-2002, Sep 13 2002
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
    
    def remove_word_url(self, text):
        return re.sub(r'\burl\b', '', text, flags=re.IGNORECASE)
    
    def remove_word_original_message(self, text):
        return re.sub(r'\boriginal message\b', '', text, flags=re.IGNORECASE)
    
    def remove_single_characters(self, text):
        # Remove single characters that are not part of a word
        return re.sub(r'\b\w\b', '', text)
    
    def remove_nt_variations(self, text):
        return re.sub(r'\bnt+ts?\b', '', text)
    
    def remove_repetitive_patterns(self, text):
        # Remove repetitive 'n' and 'nt' patterns
        return re.sub(r'\b(n|nt)+\b', '', text)
    
    def remove_repetitive_patterns(self, text):
        # Remove repetitive 't', 'n', and 'nt' patterns
        return re.sub(r'\b(t+|n+|nt+)\b', '', text)
    
    def lowercase_text(self, text):
        return text.lower()
    
    def remove_bullet_points_and_symbols(self, text):
        # List of bullet points and similar symbols
        symbols = ['•', '◦', '◉', '▪', '▫', '●', '□', '■', '✦', '✧', '✪', '✫', '✬', '✭', '✮', '✯', '✰']
        
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
                text = self.remove_word_url(text)
                text = self.remove_word_original_message(text)
                text = self.remove_single_characters(text)
                text = self.remove_nt_variations(text)
                text = self.remove_repetitive_patterns(text)
                text = self.lowercase_text(text)
                text = self.remove_bullet_points_and_symbols(text)
                text = self.remove_repeating_characters(text)
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

# Plot word clouds
def plot_word_cloud(text_list, title, width=1500, height=1000, background_color='white', max_words=300, stopwords=None, colormap='viridis', save_to_file=None):
    logging.info(f"Generating word cloud for {title}...")
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

# Combine BERT features with metadata features
def combine_features(bert_features, df_clean, headers_df, preprocessor):
    
    bert_features_np = np.array(bert_features)
    
    # Convert features to a DataFrame
    bert_features_df = pd.DataFrame(bert_features_np)
    
    # Add the labels to the BERT features DataFrame for alignment
    bert_features_df['label'] = df_clean['label'].reset_index(drop=True)
    
    # Logging BERT feature details
    logging.info("BERT Features Shape: %s", bert_features_np.shape)
    logging.info("BERT Features Type: %s", type(bert_features))
    
    # Convert lists of URLs to a single string in the 'Links' column
    headers_df['Links'] = headers_df['Links'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    # Apply the preprocessor to the metadata columns: 'From', 'To', 'Mail-To', and 'Links'
    X_metadata = headers_df[['From', 'To', 'Mail-To', 'Links']]  # Metadata columns
    X_combined = preprocessor.fit_transform(X_metadata)  # Apply transformation
    
    # Logging metadata details
    logging.info(f"Metadata Shape: {X_metadata.shape}")
    # Logging combined data details
    logging.info(f"X_combined Shape: {X_combined.shape}\n")
    
    # Convert sparse matrix to DataFrame
    X_combined_df = pd.DataFrame(X_combined.toarray())

    # Ensure indices match
    bert_features_df = bert_features_df.reset_index(drop=True)
    X_combined_df = X_combined_df.reset_index(drop=True)
    
    # Merge BERT features with metadata features
    X_final = pd.concat([bert_features_df, X_combined_df], axis=1)  # Merge BERT and metadata features

    return X_final
    
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
        input_ids = inputs['input_ids'].squeeze(dim=0)  # Ensure that the dimensions are correctly handled
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

# Split the data into training and testing sets
def split_data(features_df, test_size=0.2, random_state=42):
    logging.info("Splitting the data into training and testing sets...\n")
    # Assuming 'label' is the column name for labels in features_df
    X = features_df.drop(columns=['label'])
    y = features_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Handle data imbalance
def handle_data_imbalance(X_train, y_train, random_state=42):
    logging.info("Handling data imbalance...\n")
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
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=4)
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
    y_train_pred = ensemble_model.predict(X_train_balanced)  # Predictions on the training set
    y_test_pred = ensemble_model.predict(X_test)    # Predictions on the test set
    
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Phishing', 'Safe']
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    #print("Classification Report:\n", classification_report(y_test, y_test_pred))
    print("Classification Report for Training Data:")
    print(classification_report(y_train_balanced, y_train_pred, target_names=target_names))
    # Print classification report for test data
    print("\nClassification Report for Test Data:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))


# Main processing function
def main():
    # Use relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_email_file = os.path.join(
        base_dir, 'Extracted Data', 'extracted_email_info.csv')
    clean_email_file = os.path.join(
        base_dir, 'Extracted Data', 'clean_email_info.csv')
    merged_file =  os.path.join(
        base_dir, 'Extracted Data', 'merged_file.csv')

    dataset = load_dataset('talby/spamassassin',
                           split='train', trust_remote_code=True)
    df = dataset.to_pandas()

    try:
        # Check for missing and duplicate values
        check_missing_values = df.isnull().sum()
        logging.info(f"Check missing values:\n{check_missing_values}\n")
        
        # Check for duplicate values and remove them
        df_remove_duplicate = remove_duplicate(df)
        logging.info(f"Total number of rows remaining in the cleaned DataFrame: {df_remove_duplicate.shape[0]}\n")
        logging.debug(f"DataFrame after removing duplicates:\n{df_remove_duplicate.head()}\n")

        # Visualize data before and after removing duplicate
        visualize_data_comparison(df, df_remove_duplicate)

        # Extract email header information
        header_extractor = EmailHeaderExtractor(df_remove_duplicate)
        headers_df = header_extractor.extract_headers()
        header_extractor.save_to_csv(extracted_email_file)
        logging.info("Email header extraction and saving completed.\n")

        # Text processing (Text only)
        processor = TextProcessor()
        df_clean = processor.transform(df_remove_duplicate['text'], df_remove_duplicate['label'])
        processor.save_to_csv_cleaned(df_clean, clean_email_file)
        logging.info("Text processing and saving completed.\n")
        
        # Plot word clouds 
        #plot_word_cloud(df_remove_duplicate['text'], "Original Dataset")
        plot_word_cloud(df_clean['cleaned_text'], "Cleaned Dataset")
        
        # Feature extraction using BERT
        feature_extractor = BERTFeatureExtractor()
        texts = df_clean['cleaned_text'].tolist()
        bert_features = feature_extractor.extract_features(texts)
        
        preprocessor = ColumnTransformer(
        transformers=[
            ('email_fields', OneHotEncoder(handle_unknown='ignore'), ['From', 'To', 'Mail-To']),  # One-hot encode 'from', 'to', and 'mail_to'
            ('links', TfidfVectorizer(), 'Links')  # Vectorize links in the email
        ])
        
        # Combine features from BERT and metadata
        X_final = combine_features(bert_features, df_clean, headers_df, preprocessor)
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(X_final)
        
        # Handle data imbalance
        X_train_balanced, y_train_balanced = handle_data_imbalance(X_train, y_train)
        
        # Train and evaluate the ensemble model
        train_and_evaluate_ensemble(X_train_balanced, y_train_balanced, X_test, y_test)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    main()
