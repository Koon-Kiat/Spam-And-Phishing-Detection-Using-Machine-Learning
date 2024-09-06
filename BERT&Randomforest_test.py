# Description: This file is used to test the data cleaning and processing functions.
# Import necessary libraries

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

# String and regular expression operations
import string  # String operations
import re  # Regular expressions

# HTML and XML parsing
from bs4 import BeautifulSoup  # HTML and XML parsing

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

# Warnings
import warnings  # Warning control

# Datasets
from datasets import load_dataset  # Load datasets

# Import necessary libraries for the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Example classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
from typing import Union
from tqdm import tqdm  # Import tqdm for progress bars
warnings.filterwarnings('ignore')

# ANSI escape codes for text formatting
BOLD = '\033[1m'
RESET = '\033[0m'

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Remove duplicate data
def remove_duplicate(df):
    num_duplicates_before = df.duplicated(subset=['text'], keep=False).sum()
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
    num_duplicates_after = df_cleaned.duplicated(subset=['text'], keep=False).sum()
    duplicates_removed = num_duplicates_before - num_duplicates_after

    logging.info(f"Total number of rows identified as duplicates based on 'text': {num_duplicates_before}")
    logging.info(f"Number of rows removed due to duplication: {duplicates_removed}")

    return df_cleaned

# Visualize data
def visualize_data(df):
    label_map = {1: 'Ham', 0: 'Spam'}
    label_counts = df['label'].value_counts()
    ham_count = label_counts.get(1, 0)
    spam_count = label_counts.get(0, 0)
    total_count = ham_count + spam_count

    if total_count == 0:
        logging.warning("No data to visualize.")
        return
    
    # Plot distribution of ham and spam emails
    data = [ham_count / total_count, spam_count / total_count]
    labels = ['Ham', 'Spam']
    colors = ['blue', 'red']
    
    #
    plt.figure(figsize=(12, 5))
    wedges, texts, autotexts = plt.pie(data, labels=labels, autopct='%.0f%%', colors=colors, startangle=140)
    plt.title('Distribution of Ham and Spam Emails')

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        plt.text(x, y, f'{labels[i]}: {label_counts.get(i, 0)}', ha='center', va='center', fontsize=12, color='black')

    plt.show()
    
    # Plot count of ham and spam emails
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='label', data=df, palette=colors, order=[1, 0])
    plt.title('Ham vs Spam Email Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height / total_count:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

# Extract email information
class EmailInformationExtractor:
    def __init__(self, input_data: Union[str, pd.DataFrame], num_workers: int = 3):
        self.input_data = input_data
        self.num_workers = num_workers

    def extract_email_info(self) -> pd.DataFrame:
        if isinstance(self.input_data, str):
            df = pd.read_csv(self.input_data)
        elif isinstance(self.input_data, pd.DataFrame):
            df = self.input_data
        else:
            raise ValueError("input_data should be either a file path or a DataFrame.")

        # Initialize DataFrame to store extracted information
        extracted_df = pd.DataFrame(columns=[
            'From', 'To', 'Delivered-To', 'Subject', 'Reply-To', 'Content-Type',
            'Return-Path', 'Received', 'Message-ID', 'MIME', 'DKIM-Signature',
            'Authentication-Results', 'Links', 'Keywords', 'label'
        ])

        email_texts = df['text'].values.tolist() # Extract email texts
        labels = df['label'].values.tolist() # Extract labels

        num_cores = os.cpu_count()
        logging.info(f"Number of available CPU cores: {num_cores}")
        logging.info(f"Number of threads used for extraction: {self.num_workers}")

        # Extract email information using multithreading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(executor.map(self._extract_info_from_text, email_texts), total=len(email_texts), desc="Extracting email info"))

        # Add labels to extracted information
        for i, result in enumerate(results):
            result['label'] = labels[i]

        extracted_df = pd.concat([extracted_df, pd.DataFrame(results)], ignore_index=True)

        return extracted_df

    def _extract_info_from_text(self, text: str) -> Dict[str, str]:
        info = {
            'From': self._extract_header(text, 'From:'),
            'To': self._extract_header(text, 'To:'),
            'Delivered-To': self._extract_header(text, 'Delivered-To:'),
            'Subject': self._extract_header(text, 'Subject:'),
            'Reply-To': self._extract_header(text, 'Reply-To:'),
            'Content-Type': self._extract_header(text, 'Content-Type:'),
            'Return-Path': self._extract_header(text, 'Return-Path:'),
            'Received': self._extract_header(text, 'Received:'),
            'Message-ID': self._extract_header(text, 'Message-ID:'),
            'MIME': self._extract_header(text, 'MIME-Version:'),
            'DKIM-Signature': self._extract_header(text, 'DKIM-Signature:'),
            'Authentication-Results': self._extract_header(text, 'Authentication-Results:'),
            'Links': self._extract_links(text),
            'Keywords': self._extract_keywords(text)
        }
        return info

    def _extract_header(self, text: str, header: str) -> str:
        pattern = re.compile(rf'{header}\s*(.*)', re.MULTILINE)
        match = pattern.search(text)
        return match.group(1).strip() if match else ''

    def _extract_links(self, text: str) -> str:
        links = re.findall(r'http[s]?://\S+', text)
        return ', '.join(links)

    def _extract_keywords(self, text: str) -> str:
        words = re.findall(r'\b\w+\b', text)
        keywords = [word.lower() for word in words if len(word) > 3]
        return ', '.join(set(keywords))

    def save_to_csv(self, output_file: str):
        try:
            df = self.extract_email_info()
            df.to_csv(output_file, index=False)
            logging.info(f"Data successfully saved to {output_file}")
            self.print_label_percentages(df)
        except Exception as e:
            logging.error(f"Error saving data to CSV: {e}")

    # Print label percentages
    def print_label_percentages(self, df: pd.DataFrame):
        label_counts = df['label'].value_counts(normalize=True) * 100
        spam_percentage = label_counts.get(1, 0)
        ham_percentage = label_counts.get(0, 0)
        logging.info(f"Percentage of spam: {spam_percentage:.2f}%")
        logging.info(f"Percentage of ham: {ham_percentage:.2f}%")

# Data cleaning
class TextProcessor(BaseEstimator, TransformerMixin):
    # Initialize TextProcessor
    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words('english')) #- {'no', 'not', 'nor'}
        self.lemmatizer = WordNetLemmatizer() # Lemmatizer
        self.spell_checker = SpellChecker() # Spell checker
        self.common_words = set(self.spell_checker.word_frequency.keys()) # Common words
        self.enable_spell_check = enable_spell_check # Enable spell check

    def expand_contractions(self, text):
        return contractions.fix(text)

    def clean_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def to_lowercase(self, text):
        return text.lower()

    def remove_urls(self, text):
        return re.sub(r'https?://\S+', '', text)

    def remove_emails(self, text):
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    def remove_extra_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_digits(self, text):
        return re.sub(r'\d+', '', text)

    def remove_non_ascii(self, text):
        return re.sub(r'[^\x00-\x7F]+', ' ', text)

    def remove_single_characters(self, words_list):
        return [word for word in words_list if len(word) > 1]

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, words_list):
        return [w for w in words_list if w not in self.stop_words]

    def lemmatize(self, words_list):
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    # Cached spell check
    @lru_cache(maxsize=10000)
    def cached_spell_check(self, word):
        # Skip common words and None values
        if word is None or word in self.common_words:
            return word
        corrected_word = self.spell_checker.correction(word)
        return corrected_word if corrected_word is not None else word

    def correct_spelling(self, words_list):
        # Correct spelling using multithreading
        with ThreadPoolExecutor() as executor:
            future_to_word = {executor.submit(self.cached_spell_check, word): word for word in words_list}
            corrected_words = []
            for future in as_completed(future_to_word):
                corrected_words.append(future.result())
        return corrected_words

    def fit(self, X, y=None):
        return self

    # Transform text
    def transform(self, X, y=None):
        cleaned_text_list = []

        for body in tqdm(X, desc='Cleaning Text', unit='email'):
            try:
                text = self.clean_html(body)
                text = self.expand_contractions(text)
                text = self.to_lowercase(text)
                text = self.remove_urls(text)
                text = self.remove_emails(text)
                text = self.remove_extra_whitespace(text)
                text = self.remove_punctuation(text)
                text = self.remove_digits(text)
                text = self.remove_non_ascii(text)
                words_list = self.tokenize(text)
                words_list = self.remove_stop_words(words_list)
                words_list = self.remove_single_characters(words_list)
                words_list = self.lemmatize(words_list)
                if self.enable_spell_check:
                    words_list = self.correct_spelling(words_list)
                cleaned_text_list.append(' '.join(words_list))
            except Exception as e:
                logging.error(f"Error processing text: {e}")
                cleaned_text_list.append('')

        return cleaned_text_list

    def save_to_csv_cleaned(self, text_list, filename):
        try:
            df = pd.DataFrame(text_list, columns=['cleaned_text'])
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")

# Plot word clouds
def plot_word_cloud(text_list, title, width=1000, height=500, background_color='white', max_words=200, stopwords=None, colormap='viridis', save_to_file=None):
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


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to extract features using BERT
def extract_features(texts, tokenizer, model, max_length, batch_size=16):
    dataset = TextDataset(texts, [0]*len(texts), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    features = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting BERT features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(features)

# Main function
if __name__ == "__main__":
    # Define the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Use relative paths
    extracted_email_file = os.path.join(base_dir, 'Extracted Data', 'clean_email_info.csv')
    clean_email_file = os.path.join(base_dir, 'Extracted Data', 'extracted_email_info.csv')

    dataset = load_dataset('talby/spamassassin', split='train', trust_remote_code=True)
    df = dataset.to_pandas()
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    try:
        # Check for missing and duplicate values
        check_missing_values = df.isnull().sum()
        logging.info(f"Check missing values:\n{check_missing_values}\n")
        df_remove_duplicate = remove_duplicate(df)
        logging.info(f"Total number of rows remaining in the cleaned DataFrame: {df_remove_duplicate.shape[0]}")

        # Visualize data
        visualize_data(df_remove_duplicate)

        # Extract email information
        extractor = EmailInformationExtractor(df_remove_duplicate)
        extractor.save_to_csv(extracted_email_file)

        # Text processing
        processor = TextProcessor()
        df_clean = processor.transform(df_remove_duplicate['text'])
        processor.save_to_csv_cleaned(df_clean, clean_email_file)
        logging.info("Text processing and saving completed.")

        # Machine Learning: Training a classifier with class weight 'balanced'
        
        # Features and labels
        X = df_clean  # Cleaned text features
        y = df_remove_duplicate['label']  # Labels (ham/spam)

        # Vectorize the text data
        vectorizer = TfidfVectorizer(max_features=5000)
        X_transformed = vectorizer.fit_transform(X)

        # Initialize BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Extract BERT features
        max_length = 128
        X_bert = extract_features(texts, tokenizer, model, max_length)

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Initialize the classifier with class_weight='balanced'
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

        # Create an ensemble model
        ensemble_model = VotingClassifier(estimators=[
            ('rf', rf_model),
            # Add other models here
        ], voting='soft')

        # Train the ensemble model with progress bar
        for _ in tqdm(range(1), desc="Training ensemble model"):
            ensemble_model.fit(X_train_resampled, y_train_resampled)

        # Make predictions
        y_pred = ensemble_model.predict(X_test)

        # Evaluate the model
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Plot word clouds (optional)
        plot_word_cloud(df_remove_duplicate['text'], "Original Dataset")
        plot_word_cloud(df_clean, "Cleaned Dataset")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

