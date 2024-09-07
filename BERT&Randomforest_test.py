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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # Text feature extraction
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.model_selection import train_test_split, GridSearchCV  # Model selection
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Ensemble classifiers
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # Model evaluation
from sklearn.utils import resample  # Resampling utilities
from imblearn.over_sampling import SMOTE  # Handling imbalanced data

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


# Remove duplicate data
def remove_duplicate(df):
    logging.info("Removing duplicate data...")
    num_duplicates_before = df.duplicated(subset=['text'], keep=False).sum()
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')
    num_duplicates_after = df_cleaned.duplicated(
        subset=['text'], keep=False).sum()
    duplicates_removed = num_duplicates_before - num_duplicates_after

    logging.info(f"Total number of rows identified as duplicates based on 'text': {
                 num_duplicates_before}")
    logging.info(f"Number of rows removed due to duplication: {
                 duplicates_removed}")

    return df_cleaned

# Visualize data
def visualize_data(df):
    logging.info("Visualizing data...")
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
    wedges, texts, autotexts = plt.pie(
        data, labels=labels, autopct='%.0f%%', colors=colors, startangle=140)
    plt.title('Distribution of Ham and Spam Emails')

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = wedge.r * 0.7 * np.cos(np.radians(angle))
        y = wedge.r * 0.7 * np.sin(np.radians(angle))
        plt.text(x, y, f'{labels[i]}: {label_counts.get(
            i, 0)}', ha='center', va='center', fontsize=12, color='black')

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
        ax.annotate(f'{height / total_count:.1%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

# Extract email information
class EmailInformationExtractor:
    def __init__(self, input_data: Union[str, pd.DataFrame], num_workers: int = 3):
        self.input_data = input_data
        self.num_workers = num_workers
        logging.info("Initializing EmailInformationExtractor...")

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
            'Authentication-Results', 'Links'
        ])

        email_texts = df['text'].values.tolist()  # Extract email texts
        labels = df['label'].values.tolist()  # Extract labels

        num_cores = os.cpu_count()
        logging.info(f"Extracting email information from {len(email_texts)} emails")
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
            'Links': self._extract_links(text)
        }
        return info

    def _extract_header(self, text: str, header: str) -> str:
        pattern = re.compile(rf'{header}\s*(.*)', re.MULTILINE)
        match = pattern.search(text)
        return match.group(1).strip() if match else ''

    def _extract_links(self, text: str) -> str:
        links = re.findall(r'http[s]?://\S+', text)
        return ', '.join(links)

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
        logging.info(f"Percentage of spam from extracted email information: {spam_percentage:.2f}%")
        logging.info(f"Percentage of ham from extracted email information: {ham_percentage:.2f}% \n")

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
        return [w for w in words_list if w not in self.stop_words]

    def lemmatize(self, words_list):
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    def remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def remove_emails(self, text):
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    def remove_email_headers(self, text):
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)
        return text

    def remove_word_url(self, text):
        return re.sub(r'\burl\b', '', text, flags=re.IGNORECASE)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)
    
    def remove_repeating_characters(self, text):
        return re.sub(r'(.)\1+', r'\1\1', text)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_text_list = []

        for body in tqdm(X, desc='Cleaning Text', unit='email'):
            try:
                text = self.expand_contractions(body)
                text = self.remove_email_headers(text)
                text = self.remove_urls(text)
                text = self.remove_emails(text)
                text = self.remove_word_url(text)
                text = self.remove_punctuation(text)
                text = self.remove_numbers(text)
                text = self.remove_repeating_characters(text)
                words_list = self.tokenize(text)
                words_list = self.remove_stop_words(words_list)
                lemmatized_list = self.lemmatize(words_list)

                corrected_sentences = []
                for word in lemmatized_list:
                    corrected_word = re.sub(
                        r"(?<=t)[^a-zA-Z0-9']+|[^a-zA-Z0-9']+|(?<=\w)\-|(?<=\w)\—", ' ', word)
                    corrected_word = [
                        token for token in word_tokenize(corrected_word)]
                    corrected_sentences.extend(corrected_word)

                cleaned_text_list.append(' '.join(corrected_sentences))
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
def plot_word_cloud(text_list, title, width=1000, height=500, background_color='white', max_words=200, stopwords=None, colormap='viridis', save_to_file=None):
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

# Under Review
def merge_and_process_data(df_extracted, df_clean, merged_file):
    logging.info("Merging and processing data...")
    try:
        # Concatenate the DataFrames
        df_merged = pd.concat([df_extracted, df_clean], axis=1)
        
        # Check if 'clean_label' matches 'label' and print the result
        if 'label' in df_merged.columns and 'clean_label' in df_merged.columns:
            match = df_merged['label'] == df_merged['clean_label']
            logging.info("Do 'label' and 'clean_label' columns match?")
            logging.info(match.value_counts())

        # Drop the unnecessary 'clean_label' column after concatenation
        df_merged = df_merged.drop(columns=['clean_label'])

        # Select only the 'label' and 'cleaned_text' columns
        df_merged = df_merged[['label', 'cleaned_text']]

        # Save the combined DataFrame to a new CSV file
        df_merged.to_csv(merged_file, index=False)

        total_rows = df_merged.shape[0]
        logging.info(f"Data successfully merged and saved to {merged_file}")
        logging.info(f"Total number of rows in the merged DataFrame: {total_rows}\n")

        # Calculate and print the percentage of spam and ham
        spam_percentage = (df_merged['label'].value_counts(normalize=True) * 100)[0]
        ham_percentage = (df_merged['label'].value_counts(normalize=True) * 100)[1]
        logging.info(f"Percentage of Spam (0): {spam_percentage:.2f}%")
        logging.info(f"Percentage of Ham (1): {ham_percentage:.2f}%")

        if df_merged['cleaned_text'].isnull().any():
            logging.info(f"Data contains NaN values in the 'cleaned_text' column. Filling NaN values with empty strings.\n")
            df_merged['cleaned_text'] = df_merged['cleaned_text'].fillna('')

        return df_merged

    except Exception as e:
        logging.error(f"An error occurred in merge_and_process_data: {e}")
        return None

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
        logging.info(f"Total number of rows remaining in the cleaned DataFrame: {
                     df_remove_duplicate.shape[0]}\n")
        logging.debug(f"DataFrame after removing duplicates:\n{
                      df_remove_duplicate.head()}\n")

        # Visualize data
        # visualize_data(df_remove_duplicate)

        # Extract email information
        extractor = EmailInformationExtractor(df_remove_duplicate)
        extractor.save_to_csv(extracted_email_file)

        # Text processing (Text only)
        processor = TextProcessor()
        df_clean = processor.transform(
        df_remove_duplicate['text'], df_remove_duplicate['label'])
        processor.save_to_csv_cleaned(df_clean, clean_email_file)
        logging.info("Text processing and saving completed.\n")
        
        # Feature extraction using BERT
        feature_extractor = BERTFeatureExtractor()
        texts = df_clean['cleaned_text'].tolist()
        features = feature_extractor.extract_features(texts)
        
        # Convert features to a DataFrame
        features_df = pd.DataFrame(features)
        
        # Combine features with labels
        features_df['label'] = df_clean['label'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(features_df)
        
        # Handle data imbalance
        X_train_balanced, y_train_balanced = handle_data_imbalance(X_train, y_train)
     
        # Initialize the classifier with class_weight='balanced'
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
            grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy', verbose=3, n_jobs=4)
            grid_search.fit(X_train_balanced, y_train_balanced)
            return grid_search

        # Timing the GridSearchCV
        start_time = time.time()
        best_rf_model = profile_grid_search().best_estimator_
        end_time = time.time()
        logging.info(f"GridSearchCV took {end_time - start_time:.2f} seconds")

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

        # Evaluate the model
        train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred))
      
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Call the main function
if __name__ == "__main__":
    main()
