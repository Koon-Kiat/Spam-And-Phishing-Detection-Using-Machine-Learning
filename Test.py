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

# Natural Language Toolkit
import nltk
# For lemmatizing and stemming words
from nltk.stem import PorterStemmer, WordNetLemmatizer
# A list of common stopwords (like, and, the)
from nltk.corpus import stopwords


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

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(49)

# ANSI escape code for bold text
BOLD = '\033[1m'
RESET = '\033[0m'

# Define paths to email datasets
easy_ham_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\easy_ham\easy_ham"
hard_ham_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\hard_ham\hard_ham"
spam_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\spam_2\spam_2"

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get email datasets from directory


def get_data(path):
    data = []
    # List files in the directory
    files = os.listdir(path)
    for file in files:
        # Join the path and file name
        file_path = os.path.join(path, file)
        # Open and read the file
        with open(file_path, encoding="ISO-8859-1") as f:
            email_content = f.read()
            # Append and return the data
            data.append(email_content)
    return data


class email_data_extraction:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_email_info(self, emails):
        """Extracts specific headers and additional information from a list of emails."""
        extracted_data = []

        for mail in tqdm(emails, desc="Extracting Headers"):
            b = email.message_from_string(mail)

            header_info = {
                'From': b.get('From', 'N/A'),
                'To': b.get('To', 'N/A'),
                'Delivered-To': b.get('Delivered-To', 'N/A'),
                'Subject': b.get('Subject', 'N/A'),
                'Reply-To': b.get('Reply-To', 'N/A'),
                'Content-Type': b.get('Content-Type', 'N/A'),
                'Return-Path': b.get('Return-Path', 'N/A'),
                'Received': b.get('Received', 'N/A'),
                'Message-ID': b.get('Message-ID', 'N/A'),
                'MIME': b.get('MIME-Version', 'N/A'),
                'DKIM-Signature': b.get('DKIM-Signature', 'N/A'),
                'Authentication-Results': b.get('Authentication-Results', 'N/A'),
                'Links': self.extract_links(mail)
            }
            extracted_data.append(header_info)

        # Convert list of dicts to DataFrame
        return pd.DataFrame(extracted_data)

    def extract_links(self, mail):
        """Extracts URLs from the email text."""
        b = email.message_from_string(mail)
        body = self._get_body(b)
        text = BeautifulSoup(body, "html.parser").get_text()
        links = re.findall(r'(https?://\S+)', text)
        return ', '.join(links)

    def _get_body(self, email_msg):
        """Extracts the body of the email message, handling multipart content."""
        body = ""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    body = part.get_payload(decode=True)
                    break
        else:
            body = email_msg.get_payload(decode=True)
        return body

    def save_to_csv(self, df, filename):
        """Saves the DataFrame to a CSV file."""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text_list = []
        for mail in X:
            # Converts raw email string into email message object
            b = email.message_from_string(mail)
            body = ""

            if b.is_multipart():
                for part in b.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # Skip any text/plain (txt) attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(
                            decode=True)  # get body of email
                        break
            # Not multipart - i.e. plain text, no attachments
            else:
                # Get body of email
                body = b.get_payload(decode=True)

            # Get text from body (HTML/text)
            soup = BeautifulSoup(body, "html.parser")
            # Convert text to lowercase
            text = soup.get_text().lower()
            # Remove links
            text = re.sub(
                r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
            # Remove email address
            text = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE)
            # Remove punctuation
            text = text.translate(str.maketrans(
                '', '', string.punctuation))  # remove punctuation
            # Remove digits
            text = ''.join([i for i in text if not i.isdigit()])
            # Loads a list of stopwords
            stop_words = stopwords.words('english')
            # Remove stop words
            words_list = [w for w in text.split() if w not in stop_words]
            # Lemmatize (Reduce word to base form, while retaining context)
            words_list = [lemmatizer.lemmatize(w) for w in words_list]
            # Stemming (Reduce word to base from by removing suffixes and prefixes)
            words_list = [stemmer.stem(w) for w in words_list]  # Stemming
            text_list.append(' '.join(words_list))
        return text_list


def data_visualisation_plt(ham, spam):
    data = [len(ham)/len(ham+spam), len(spam)/len(ham+spam)]
    labels = ['ham', 'spam']
    colors = ['blue', 'red']

    # Pie chart of the distribution of ham and spam
    plt.figure(figsize=(12, 5))
    plt.pie(data, labels=labels, autopct='%.0f%%', colors=colors)
    plt.show()

    # Count plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x=['ham']*len(ham) + ['spam']*len(spam), palette=colors)
    plt.show()


def data_visualisation_px(ham, spam):
    # Data for the pie chart
    data = [len(ham), len(spam)]
    labels = ['Ham', 'Spam']
    colors = ['blue', 'red']

    # Pie chart using Plotly
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=data, textinfo='label+percent', marker=dict(colors=colors))])
    fig_pie.update_layout(title='Distribution of Ham and Spam')
    fig_pie.show()

    # Data for the count plot
    categories = ['Ham'] * len(ham) + ['Spam'] * len(spam)

    # Count plot using Plotly
    fig_count = px.histogram(x=categories, color=categories, color_discrete_map={
                             'Ham': 'blue', 'Spam': 'red'})
    fig_count.update_layout(title='Count Plot of Ham and Spam',
                            xaxis_title='Category', yaxis_title='Count')
    fig_count.show()


def plot_WordCloud(text_list, title):
    unique_string = (" ").join(text_list)
    wordcloud = WordCloud(width=1000, height=500,
                          background_color='white').generate(unique_string)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()


# Get email data
easy_ham = get_data(easy_ham_path)
hard_ham = get_data(hard_ham_path)
ham = easy_ham + hard_ham
spam = get_data(spam_path)

# Shuffle email data
np.random.shuffle(ham)
np.random.shuffle(spam)

# Structure of the data
# print(spam[49])

# Run the class to extract key information from dataset
processor = email_data_extraction()
# Extract key information and save to separate files for ham and spam (No data cleaning)
hamdf = processor.extract_email_info(ham)
spamdf = processor.extract_email_info(spam)
processor.save_to_csv(hamdf, 'HamExtraction.csv')
processor.save_to_csv(spamdf, 'SpamExtraction.csv')

# Run class to clean the data set
email_to_text = email_to_clean_text()
# Process each email in ham (easy+hard) to clean the text and return a list of cleaned string)
text_ham = email_to_text.transform(ham)
print(f"{BOLD}Sample email content of HAM:{RESET}\n{text_ham[0]}")
text_spam = email_to_text.transform(spam)
print(f"{BOLD}Sample email content of SPAM:{RESET}\n{text_spam[0]}")

# For visualisation
text_easy_ham = email_to_text.transform(easy_ham)
text_hard_ham = email_to_text.transform(hard_ham)

# Shows distribution of ham and spam
data_visualisation_plt(ham, spam)

# Show the common words used in each of the datasets
plot_WordCloud(text_easy_ham, "Easy Ham")
plot_WordCloud(text_hard_ham, "Hard Ham")
plot_WordCloud(text_ham, "Ham")
plot_WordCloud(text_spam, "Spam")