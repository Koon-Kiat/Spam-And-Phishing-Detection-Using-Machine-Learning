import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the NLTK data for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset (adjust the file path as needed)
df = pd.read_csv('sample.csv')


#df_sample = df_original.sample(n=20000, random_state=42)  # Sample 20,000 rows
#df = df_sample.copy()

# Show the number of rows and missing values before cleaning
print("Original Data Summary:")
print("Total rows before sampling:", df.shape[0])
print("Missing values before cleaning:\n", df.isna().sum())


# Remove duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna(subset=['message'])

print("\nCleaned Data Summary:")
print("Total rows after cleaning:", df.shape[0])
print("Missing values after cleaning:\n", df.isna().sum())

# Define stopwords set (include only NLTK stopwords)
nltk_stopwords = set(stopwords.words('english'))
stopwords_set = nltk_stopwords

# Extract email headers, body, keywords, and links
def email_parsing(raw_message):
    email = {'body': '', 'keywords': '', 'links': ''}
    headers, body_lines = [], []
    in_headers = True

    for line in raw_message.split('\n'):
        if in_headers:
            if line.strip() == '':  # Empty line indicates end of headers
                in_headers = False
            else:
                headers.append(line.strip())
        else:
            body_lines.append(line.strip())

    email.update(parse_headers(headers))
    email['body'] = '\n'.join(body_lines).strip()
    email['keywords'] = extract_keywords(email['body'])
    email['links'] = extract_links(email['body'])
    return email

# Function to parse headers from a list of header lines


def parse_headers(header_lines):
    headers = {}
    current_header = ''
    for line in header_lines:
        if ':' in line:
            key, val = line.split(':', 1)
            headers[key.lower().strip()] = val.strip()
            current_header = key.lower().strip()
        elif current_header:
            headers[current_header] += ' ' + line.strip()
    return {'Email Header': ' | '.join([f"{k}: {v}" for k, v in headers.items()])}

# Function to extract keywords (top N words by frequency)


def extract_keywords(text, top_n=5):
    if not text:  # Check if text is empty or None
        return ''
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens if word not in stopwords_set and word.isalnum()]
    word_freq = Counter(tokens)
    most_common = word_freq.most_common(top_n)
    return ', '.join([word for word, freq in most_common])

# Function to extract links from the email body


def extract_links(text):
    if not text:  # Check if text is empty or None
        return ''
    links = re.findall(r'(https?://[^\s]+)', text)
    return ', '.join(links)

# Compile all extracted data into a single dictionary


def emails_parsing(messages):
    emails = [email_parsing(message) for message in messages]
    return {
        'Email Header': [email['Email Header'] for email in emails],
        'Links': [email['links'] for email in emails],
        'Keywords': [email['keywords'] for email in emails],
    }


# Extract the relevant data into new DataFrames
parsed_data = emails_parsing(df['message'])
email_headers_df = pd.DataFrame({'Email Header': parsed_data['Email Header']})
links_df = pd.DataFrame({'Links': parsed_data['Links']})
keywords_df = pd.DataFrame({'Keywords': parsed_data['Keywords']})

# Save data to a single CSV file
output_file = 'extracted_email_data.csv'
combined_df = pd.concat([email_headers_df, links_df, keywords_df], axis=1)
combined_df.to_csv(output_file, index=False)

print(f"Extracted data saved to {output_file}")

# Check the content of the saved file
print("Extracted DataFrame columns:", combined_df.columns)
print("Sample of combined DataFrame:")
print(combined_df.head())