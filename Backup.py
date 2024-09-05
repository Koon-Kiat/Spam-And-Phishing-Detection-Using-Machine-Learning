# Description: This file is used to test the data cleaning and processing functions.
# Import necessary libraries

# Data manipulation
import numpy as np  # For numerical operations and handling multi-dimensional arrays
import pandas as pd  # For data manipulation and analysis

# Data visualization
import matplotlib.pyplot as plt  # For creating static plots
import plotly.express as px  # For creating interactive plots
import plotly.graph_objects as go  # For more complex interactive plots
import seaborn as sns  # For statistical data visualization
from wordcloud import WordCloud  # For generating word cloud visualizations

# Operating system interfaces
import os  # For interacting with the operating system

# Email parsing
import email  # For parsing and handling email messages
import email.policy  # For handling email message policies

# String and regular expression operations
import string  # For string operations
import re  # For regular expression operations

# HTML and XML parsing
from bs4 import BeautifulSoup  # For parsing HTML and XML documents

# Progress bar
from tqdm import tqdm  # For displaying progress bars

# Logging
import logging  # For logging messages

# Text processing
import contractions  # For expanding contractions in text
import codecs  # For encoding and decoding data
import json  # For handling JSON data
import urllib.parse  # For parsing URLs

# Natural Language Toolkit (NLTK)
import nltk
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
from nltk.corpus import stopwords  # For accessing stopwords
from nltk.tokenize import word_tokenize  # For tokenizing text

# Typing support
from typing import List, Dict, Union  # For type hinting

# TextBlob
from textblob import Word  # For processing textual data

# Concurrent execution
from concurrent.futures import ThreadPoolExecutor, as_completed  # For concurrent execution
from functools import lru_cache # For caching results

# Spell checking
from spellchecker import SpellChecker  # For spell checking

# Machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin  # For creating custom transformers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # For converting text to a matrix of token counts and TF-IDF features
from sklearn.linear_model import LogisticRegression  # For logistic regression models
from sklearn.tree import DecisionTreeClassifier  # For decision tree models
from sklearn.ensemble import RandomForestClassifier  # For random forest models
from sklearn.neural_network import MLPClassifier  # For multi-layer perceptron models
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import LabelEncoder  # For encoding labels

# Deep learning libraries
from tensorflow.keras.preprocessing.text import Tokenizer  # For tokenizing text for deep learning models
from tensorflow.keras.layers import Embedding, GRU, LSTM, Bidirectional, SimpleRNN  # For various neural network layers
from tensorflow.keras.utils import pad_sequences  # For padding sequences
from keras.models import Sequential  # For creating sequential models
from keras.layers import Dense, Dropout  # For dense and dropout layers
import tensorflow as tf  # For deep learning

# Warnings
import warnings  # For handling warnings

# Datasets
import datasets  # For accessing datasets
from datasets import load_dataset  # For loading datasets

warnings.filterwarnings('ignore')

# ANSI escape codes for text formatting
BOLD = '\033[1m'
RESET = '\033[0m'

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Remove duplicate data
def remove_duplicate(df):
    # Identify duplicate rows based on 'text' column
    num_duplicates_before = df.duplicated(subset=['text'], keep=False).sum()

    # Remove duplicate rows, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
    num_duplicates_after = df_cleaned.duplicated(subset=['text'], keep=False).sum()

    # Calculate the number of duplicates removed
    duplicates_removed = num_duplicates_before - num_duplicates_after

    # Log results
    logging.info(f"Total number of rows identified as duplicates based on 'text': {num_duplicates_before}")
    logging.info(f"Number of rows removed due to duplication: {duplicates_removed}")

    return df_cleaned
    
# Visualize data
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
    wedges, texts, autotexts = plt.pie(data, labels=labels, autopct='%.0f%%', colors=colors, startangle=140)
    plt.title('Distribution of Ham and Spam Emails')
    
    # Add annotations with exact counts
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        plt.text(x, y, f'{labels[i]}: {label_counts.get(i, 0)}', ha='center', va='center', fontsize=12, color='black')
    
    plt.show()

    # Count plot
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='label', data=df, palette=colors, order=[1, 0])
    plt.title('Ham vs Spam Email Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])

    # Add percentage labels to the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height / total_count:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

# Extract email information
class EmailInformationExtractor:
    # Initialize the class with input data and number of workers
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

        # Create an empty DataFrame to store the extracted information
        extracted_df = pd.DataFrame(columns=[
            'From', 'To', 'Delivered-To', 'Subject', 'Reply-To', 'Content-Type',
            'Return-Path', 'Received', 'Message-ID', 'MIME', 'DKIM-Signature',
            'Authentication-Results', 'Links', 'Keywords', 'label'
        ])

        email_texts = df['text'].values.tolist() # Extract email texts
        labels = df['label'].values.tolist()  # Extract labels

        # Log the number of available CPU cores and threads used
        num_cores = os.cpu_count()
        logging.info(f"Number of available CPU cores: {num_cores}")
        logging.info(f"Number of threads used for extraction: {self.num_workers}")
        
        # Extract email information using multiple threads
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(executor.map(self._extract_info_from_text, email_texts), total=len(email_texts), desc="Extracting email info"))

        # Add labels to the results
        for i, result in enumerate(results):
            result['label'] = labels[i]
            
        # Append the extracted information to the DataFrame
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
        # Calculate label percentages
        label_counts = df['label'].value_counts(normalize=True) * 100
    
        # Map numeric labels to 'spam' and 'ham'
        spam_percentage = label_counts.get(1, 0)
        ham_percentage = label_counts.get(0, 0)
    
        # Log the percentages of spam and ham
        logging.info(f"Percentage of spam: {spam_percentage:.2f}%")
        logging.info(f"Percentage of ham: {ham_percentage:.2f}%")
        
# Data cleaning
class TextProcessor(BaseEstimator, TransformerMixin):
    # Initialize the class with the necessary resources
    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.common_words = set(self.spell_checker.word_frequency.keys())
        self.enable_spell_check = enable_spell_check

    def expand_contractions(self, text):
        text = contractions.fix(text)
        return text

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

    @lru_cache(maxsize=10000)
    def cached_spell_check(self, word):
        if word is None or word in self.common_words:
            return word
        corrected_word = self.spell_checker.correction(word)
        return corrected_word if corrected_word is not None else word

    def correct_spelling(self, words_list):
        with ThreadPoolExecutor() as executor:
            future_to_word = {executor.submit(self.cached_spell_check, word): word for word in words_list}
            corrected_words = []
            for future in as_completed(future_to_word):
                corrected_words.append(future.result())
        return corrected_words

    def fit(self, X, y=None):
        return self

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
    # Combine all text into a single string
    unique_string = " ".join(text_list)

    # Generate the word cloud
    wordcloud = WordCloud(width=width, height=height,
                          background_color=background_color,
                          max_words=max_words,
                          stopwords=stopwords,
                          colormap=colormap).generate(unique_string)

    # Plot the word cloud
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()

    # Save the word cloud image to a file if specified
    if save_to_file:
        wordcloud.to_file(save_to_file)
        print(f"Word cloud saved to {save_to_file}")

# Main function
if __name__ == "__main__": 
    extracted_email_file = 'extracted_email_info.csv'
    clean_email_file = 'clean_email_info.csv'
    
     # Load the dataset
    dataset = load_dataset('talby/spamassassin', split='train')

    # Assuming you want to view the first few rows of the 'train' split
    df = dataset.to_pandas()

    try:
        # Check for missing values
        check_missing_values = df.isnull().sum()
        print(f"\nCheck missing values:\n{check_missing_values}\n")

        df_original_text = df['text']

        # Remove duplicate
        df_remove_duplicate = remove_duplicate(df)
        logging.info(f"Total number of rows remaining in the cleaned DataFrame: {df_remove_duplicate.shape[0]}")

        #visualize_data(df_remove_duplicate)

        # Extract important information from email
        #extractor = EmailInformationExtractor(df_remove_duplicate)
        #extractor.save_to_csv(extracted_email_file)

        # Perform data cleaning
        processor = TextProcessor()
        df_clean = processor.transform(df_remove_duplicate['text'])
        processor.save_to_csv_cleaned(df_clean, clean_email_file)
        logging.info("Text processing and saving completed.")
        
        # Plot word clouds
        #plot_word_cloud(df_original_text, "Original Dataset")
        #plot_word_cloud(df_clean, "Cleaned Dataset")

    except Exception as e:
        logging.error(f"An error occurred: {e}")