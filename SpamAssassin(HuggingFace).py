# Linear algebra
import numpy as np

# Data processing
import pandas as pd

# Plotting graph
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
# Visual representation of word frequency or importance
from wordcloud import WordCloud

# Interact with operating system
import os
# Manage email message
import email
import email.policy
# String formatting and manipulation etc.
import string
# Provide regex
import re
# Parsing HTML and XML documents
from bs4 import BeautifulSoup
# Progress bar library
from tqdm import tqdm
# Log message
import logging

import contractions
import codecs
import json
import urllib.parse

# Natural Language Toolkit
import nltk
# For lemmatizing and stemming words
from nltk.stem import PorterStemmer, WordNetLemmatizer
# A list of common stopwords (like, and, the)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from typing import List, Dict, Union
from textblob import Word
from concurrent.futures import ThreadPoolExecutor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, LSTM, Bidirectional, SimpleRNN
from tensorflow.keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import warnings

import datasets
from datasets import load_dataset

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(49)

# ANSI escape code for bold text
BOLD = '\033[1m'
RESET = '\033[0m'

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
dataset = load_dataset('talby/spamassassin', split='train')

# Assuming you want to view the first few rows of the 'train' split
df = dataset.to_pandas()

check_missing_values = df.isnull().sum()
print(f"\nCheck missing values:\n{check_missing_values}\n")

# Remove duplicate data
def remove_duplicate(df):
    # Identify duplicate rows
    duplicate_text = df[df.duplicated(subset=['text'], keep=False)]

    # Count of duplicates before dropping
    num_duplicates_before = duplicate_text.shape[0]

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
    num_duplicates_after = df_cleaned.shape[0]

    # Count of duplicates after dropping
    remaining_duplicate = num_duplicates_before - num_duplicates_after

    # Print results
    print("\nTotal number of rows identified as duplicates based on 'text':",
          num_duplicates_before)
    print("Number of rows removed due to duplication:", remaining_duplicate)

    return df_cleaned

# Plot graph to visualise distribution of data across ham and spam using matplotlib.pyplot
def visualize_data(df):

    # Map class labels to meaningful names
    label_map = {1: 'Ham', 0: 'Spam'}

    # Count the number of ham and spam emails
    label_counts = df['label'].value_counts()

    # Extract counts for ham (1) and spam (0) with default value of 0 if missing
    ham_count = label_counts.get(1, 0)
    spam_count = label_counts.get(0, 0)

    # Calculate total count and check for zero division
    total_count = ham_count + spam_count
    if total_count == 0:
        print("No data to visualize.")
        return

    # Calculate distribution percentages
    data = [ham_count / total_count, spam_count / total_count]
    labels = ['Ham', 'Spam']
    colors = ['blue', 'red']

    # Pie chart
    plt.figure(figsize=(12, 5))
    plt.pie(data, labels=labels, autopct='%.0f%%', colors=colors)
    plt.title('Distribution of Ham and Spam Emails')
    plt.show()

    # Count plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df, palette=colors, order=[1, 0])
    plt.title('Ham vs Spam Email Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
    plt.show()

# Extract information from data without cleaning
class email_information_extractor:
    def __init__(self, input_data: Union[str, pd.DataFrame], num_workers: int = 3):
        self.input_data = input_data
        self.num_workers = num_workers

    def extract_email_info(self) -> pd.DataFrame:
        if isinstance(self.input_data, str):
            df = pd.read_csv(self.input_data)
        elif isinstance(self.input_data, pd.DataFrame):
            df = self.input_data
        else:
            raise ValueError(
                "input_data should be either a file path or a DataFrame.")

        extracted_df = pd.DataFrame(columns=[
            'From', 'To', 'Delivered-To', 'Subject', 'Reply-To', 'Content-Type',
            'Return-Path', 'Received', 'Message-ID', 'MIME', 'DKIM-Signature',
            'Authentication-Results', 'Links', 'Keywords'
        ])

        email_texts = df['text'].values.tolist()

        # Print the number of available CPU cores and threads used
        num_cores = os.cpu_count()
        print(f"Number of available CPU cores: {num_cores}")
        print(f"Number of threads used for extraction: {self.num_workers}")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(executor.map(self._extract_info_from_text, email_texts), total=len(
                email_texts), desc="Extracting email info"))

        extracted_df = pd.concat(
            [extracted_df, pd.DataFrame(results)], ignore_index=True)

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
        df = self.extract_email_info()
        df.to_csv(output_file, index=False)

# Data cleaning
class email_text_cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize NLTK tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add additional custom stop words if necessary
        self.custom_stop_words = {'url', 'http', 'https', 'www'}
        self.stop_words.update(self.custom_stop_words)

    def fit(self, X, y=None):
        # No fitting is needed for text cleaning
        return self
    
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    def correct_spelling(self, text):
        return ' '.join([str(Word(word).correct()) for word in text.split()])

    def transform(self, X, y=None):
        cleaned_text_list = []

    def expand_contractions(self, text):
        return contractions.fix(text)
    
    def correct_spelling(self, text):
        # Use textblob to correct spelling
        return ' '.join([str(Word(word).correct()) for word in text.split()])

    def transform(self, X, y=None):
        cleaned_text_list = []

        # Initialize tqdm progress bar
        for body in tqdm(X, desc='Cleaning Text', unit='email'):
            # Convert HTML to plain text
            soup = BeautifulSoup(body, "html.parser")
            text = soup.get_text()

            # Expand contractions
            text = self.expand_contractions(text)

            # Convert to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)

            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

            # Remove multiple newlines and extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove digits
            text = re.sub(r'\d+', '', text)

            # Tokenization
            words_list = word_tokenize(text)

            # Remove stop words
            words_list = [w for w in words_list if w not in self.stop_words]

            # Correct spelling errors (optional, depending on performance)
            # words_list = [str(Word(word).correct()) for word in words_list]

            # Lemmatization
            words_list = [self.lemmatizer.lemmatize(w) for w in words_list]

            # Join cleaned words into a single string
            cleaned_text_list.append(' '.join(words_list))

        return cleaned_text_list
            

    def save_to_csv_cleaned(self, text_list, filename):
        df = pd.DataFrame(text_list, columns=['cleaned_text'])
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

# Show word frequency
def plot_word_cloud(text_list, title):
    # Combine all text into a single string
    unique_string = " ".join(text_list)

    # Generate the word cloud
    wordcloud = WordCloud(width=1000, height=500,
                          background_color='white').generate(unique_string)

    # Plot the word cloud
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()


if __name__ == "__main__":
    extracted_email_file = 'extracted_email_info.csv'
    clean_email_file = 'clean_email_info.csv'

    df_original_text = df['text']

    # Remove duplicate
    df_remove_duplicate = remove_duplicate(df)
    print("Total number of rows remaining in the cleaned DataFrame:",
          df_remove_duplicate.shape[0])

    # Compare amount of data before and after removing duplicates
    visualize_data(df_remove_duplicate)

    # Converts to dataframe
    df_remove_duplicate = pd.DataFrame(df_remove_duplicate)

    # Extract important information from email
    extractor = email_information_extractor(df_remove_duplicate)
    extractor.save_to_csv(extracted_email_file)

    # Peforms data cleaning
    clean_email = email_text_cleaner()
    df_clean = clean_email.transform(df_remove_duplicate['text'])
    clean_email.save_to_csv_cleaned(df_clean, clean_email_file)

    plot_word_cloud(df_original_text, "Original Dataset")
    plot_word_cloud(df_clean, "Cleaned Dataset")
