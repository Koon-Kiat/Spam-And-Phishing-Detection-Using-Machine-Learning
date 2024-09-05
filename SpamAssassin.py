import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import email
import string
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import warnings
from tqdm import tqdm  # Progress bar library
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(49)

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define paths to email datasets
easy_ham_path = r"filepath"
hard_ham_path = r"filepath"
spam_path = r"filepath"

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_data(path):
    """Loads email data from a specified directory."""
    data = []
    files = os.listdir(path)
    for file in files:
        # Join the path and file name
        file_path = os.path.join(path, file)
        with open(file_path, encoding="ISO-8859-1") as f:
            email_content = f.read()
            data.append(email_content)
    return data

def clean_data(df):
    """Performs data cleaning on the email dataset."""
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values
    df.fillna("N/A", inplace=True)
    
    # Normalize text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].str.lower().str.strip()
    
    return df

def extract_headers(emails):
    """Extracts specific headers from a list of emails and additional information."""
    extracted_data = []
    for mail in tqdm(emails, desc="Extracting Headers"):
        # Parse email
        b = email.message_from_string(mail)
        
        # Extract headers
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
            'SPF': b.get('Authentication-Results', 'N/A'),
            'Keywords': extract_keywords(mail),  # Extracted keywords
            'Links': extract_links(mail)         # Extracted links
        }
        extracted_data.append(header_info)
    
    return pd.DataFrame(extracted_data)

def extract_keywords(mail):
    """Extracts keywords from email text."""
    b = email.message_from_string(mail)
    body = ""
    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)
                break
    else:
        body = b.get_payload(decode=True)
    
    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text().lower()
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    
    stop_words = stopwords.words('english')
    words_list = [w for w in text.split() if w not in stop_words]
    words_list = [lemmatizer.lemmatize(w) for w in words_list]
    words_list = [stemmer.stem(w) for w in words_list]
    
    return ' '.join(words_list)

def extract_links(mail):
    """Extracts links from email text."""
    b = email.message_from_string(mail)
    body = ""
    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)
                break
    else:
        body = b.get_payload(decode=True)
    
    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text()
    links = re.findall(r'(https?://\S+)', text)
    
    return ', '.join(links)

def save_to_csv(df, filename):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def visualize_word_frequencies(texts, title):
    """Visualizes word frequencies using a bar chart."""
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    word_freq = np.asarray(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    
    word_freq_df = pd.DataFrame({'Word': words, 'Frequency': word_freq})
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=word_freq_df, palette='viridis')
    plt.title(title)
    plt.show()

class EmailToCleanText(BaseEstimator, TransformerMixin):
    """Custom transformer to clean email text data."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transforms raw email data into clean text."""
        text_list = []
        for mail in tqdm(X, desc="Cleaning Emails"):
            # Parse email content
            email_message = email.message_from_string(mail)
            body = ""

            # Handle multipart emails
            if email_message.is_multipart():
                for part in email_message.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # Skip attachments, focus on text/plain
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(decode=True)  # get body of email
                        break
            else:
                body = email_message.get_payload(decode=True)  # get body of email
            
            # Extract text from HTML content
            soup = BeautifulSoup(body, "html.parser")
            text = soup.get_text().lower()
            
            # Remove URLs
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove digits
            text = ''.join([i for i in text if not i.isdigit()])
            
            # Remove stop words
            stop_words = stopwords.words('english')
            words_list = [word for word in text.split() if word not in stop_words]
            
            # Lemmatization
            words_list = [lemmatizer.lemmatize(word) for word in words_list]
            
            # Stemming
            words_list = [stemmer.stem(word) for word in words_list]
            
            text_list.append(' '.join(words_list))
        
        return text_list

def visualize_data(ham, spam):
    """Visualizes the distribution of ham and spam emails."""
    data = [len(ham)/len(ham + spam), len(spam)/len(ham + spam)]
    labels = ['ham', 'spam']
    colors = ['green', 'red']
    
    # Pie chart
    plt.figure(figsize=(12, 5))
    plt.pie(data, labels=labels, autopct='%.0f%%', colors=colors)
    plt.title('Distribution of Ham and Spam Emails')
    plt.show()
    
    # Count plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x=['ham']*len(ham) + ['spam']*len(spam), palette=colors)
    plt.title('Ham vs Spam Email Count')
    plt.show()

def plot_wordcloud(text_list, title):
    """Generates and plots a word cloud."""
    unique_string = " ".join(text_list)
    wordcloud = WordCloud(width=1000, height=500).generate(unique_string)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Load email data
easy_ham = get_data(easy_ham_path)
hard_ham = get_data(hard_ham_path)
spam = get_data(spam_path)

# Shuffle the data
np.random.shuffle(easy_ham)
np.random.shuffle(hard_ham)
np.random.shuffle(spam)

# Combine ham datasets
ham = easy_ham + hard_ham

# Transform email content into clean text
email_to_text = EmailToCleanText()
text_ham = email_to_text.transform(ham)
text_spam = email_to_text.transform(spam)

# Compare number of emails before and after cleaning
print(f"Number of ham emails before cleaning: {len(ham)}")
print(f"Number of spam emails before cleaning: {len(spam)}")

# Create DataFrame of email headers and additional information
headers_df = extract_headers(ham + spam)

# Save the DataFrame to CSV
save_to_csv(headers_df, 'cleaned_email_headers.csv')

# Visualize the data
visualize_data(ham, spam)

# Generate word clouds
plot_wordcloud(email_to_text.transform(easy_ham), "Easy Ham Word Cloud")
plot_wordcloud(email_to_text.transform(hard_ham), "Hard Ham Word Cloud")
plot_wordcloud(text_spam, "Spam Word Cloud")

# Visualize word frequencies before and after cleaning
visualize_word_frequencies(ham, "Word Frequencies Before Cleaning")
visualize_word_frequencies(text_ham + text_spam, "Word Frequencies After Cleaning")
