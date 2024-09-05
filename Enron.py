import pandas as pd
import numpy as np
import nltk
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from email import message_from_string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Define global variables for column headers
MESSAGE_COLUMN = 'message'
NORMALIZED_MESSAGE_COLUMN = 'normalized_message'
LINKS_COLUMN = 'links'
KEYWORDS_COLUMN = 'keywords'

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    #print("Loading dataset...")
    readfile = pd.read_csv(file_path)
    readfile = readfile.sample(n=55000, random_state=42)
    return readfile

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    #print("Removing duplicates...")
    initial_count = df.shape[0]
    df_cleaned = df.drop_duplicates()
    final_count = df_cleaned.shape[0]
    print(f"Removed {initial_count - final_count} duplicate rows.")
    return df_cleaned

def handle_missing_values(df):
    """Handle missing values by dropping rows with missing messages."""
    #print("Handling missing values...")
    df = df.dropna(subset=[MESSAGE_COLUMN])
    # Replace any remaining missing values with 'NA'
    df.fillna('NA', inplace=True)
    return df

def normalize_text(text):
    """Normalize the text by lowercasing, removing stopwords, and stemming."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def normalize_text_worker(text):
    """Worker function for multiprocessing."""
    return normalize_text(text)

def normalize_text_data(df, num_cores=2):
    """Apply normalization to the text data using a specified number of CPU cores."""
    #print("Normalizing text data...")
    
    # Check the number of cores being used
    check_cores(num_cores)

    # Initialize the pool of workers
    pool = Pool(num_cores)
    df[NORMALIZED_MESSAGE_COLUMN] = list(tqdm(pool.imap(normalize_text_worker, df[MESSAGE_COLUMN]), total=len(df)))
    pool.close()
    pool.join()
    
    return df

def extract_links(text):
    """Extract all URLs from the given text."""
    return re.findall(r'http[s]?://\S+', text)

def extract_keywords(text):
    """Extract keywords from the normalized text."""
    tokens = word_tokenize(text)
    keywords = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ', '.join(keywords)

def parse_headers(text):
    """Parse email headers from the given text."""
    msg = message_from_string(text)
    headers = {
        'From': msg.get('From', 'NA'),
        'To': msg.get('To', 'NA'),
        'Delivered-To': msg.get('Delivered-To', 'NA'),
        'Subject': msg.get('Subject', 'NA'),
        'Reply-To': msg.get('Reply-To', 'NA'),
        'Content-Type': msg.get('Content-Type', 'NA'),
        'Return-Path': msg.get('Return-Path', 'NA'),
        'Received': msg.get('Received', 'NA'),
        'Message-ID': msg.get('Message-ID', 'NA'),
        'MIME-Version': msg.get('MIME-Version', 'NA'),
        'DKIM-Signature': msg.get('DKIM-Signature', 'NA'),
        'SPF': msg.get('SPF', 'NA')
    }
    return pd.Series(headers)

def extract_data(df):
    """Extract email headers, links, and keywords from the dataset."""
    #print("Extracting data...")
    
    # Extract headers
    header_df = df[MESSAGE_COLUMN].apply(parse_headers)
    
    # Extract links and keywords
    df[LINKS_COLUMN] = df[MESSAGE_COLUMN].apply(lambda x: ', '.join(extract_links(x)))
    df[KEYWORDS_COLUMN] = df[NORMALIZED_MESSAGE_COLUMN].apply(extract_keywords)
    
    # Combine headers with features
    result_df = pd.concat([header_df, df[[LINKS_COLUMN, KEYWORDS_COLUMN]]], axis=1)
    
    print("Data extraction complete.")
    return result_df

def visualize_word_frequencies(df, title_suffix=''):
    """Plot word frequencies from the DataFrame."""
    # Compute word frequencies
    all_words = ' '.join(df[MESSAGE_COLUMN] if MESSAGE_COLUMN in df.columns else df[NORMALIZED_MESSAGE_COLUMN])
    tokens = word_tokenize(all_words)
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    word_freq = nltk.FreqDist(words)
    
    # Get the top 20 words
    top_words = word_freq.most_common(20)
    words, frequencies = zip(*top_words)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(frequencies), y=list(words), palette='viridis')
    plt.title(f'Top 20 Words Frequency {title_suffix}')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()

def visualize_cleaning_process(original_size, after_duplicates_size, after_missing_values_size, after_normalization_size):
    """Visualize the data cleaning process showing the number of records before and after each cleaning step."""
    print("Visualizing the data cleaning process...")

    # Data for plotting
    stages = [
        'Original Data',
        'After Removing Duplicates',
        'After Handling Missing Values',
        'After Normalization'
    ]
    sizes = [original_size, after_duplicates_size, after_missing_values_size, after_normalization_size]

    # Convert stages to DataFrame for plotting
    df_plot = pd.DataFrame({
        'Stage': stages,
        'Size': sizes
    })

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Stage', y='Size', data=df_plot, palette='coolwarm', hue='Stage', legend=False)
    plt.title('Dataset Size at Each Cleaning Stage')
    plt.xlabel('Cleaning Stages')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45)
    plt.show()

def compare_data(original_df, cleaned_df):
    """Compare the original and cleaned datasets."""
    print("Comparing cleaned data with original...")
    original_size = original_df.shape[0]
    cleaned_size = cleaned_df.shape[0]
    print(f"Original dataset size: {original_size}")
    print(f"Cleaned dataset size: {cleaned_size}")
    print(f"Number of removed rows: {original_size - cleaned_size}")

def save_cleaned_data(df, output_file):
    """Save the cleaned data to a CSV file."""
    #print("Saving cleaned data to CSV file...")
    df.to_csv(output_file, index=False)
    print("Data extraction complete.")

def check_cores(num_cores):
    """Print the number of available and used CPU cores."""
    total_cores = cpu_count()
    print(f"Total available CPU cores: {total_cores}")
    print(f"Number of cores being used: {num_cores}")

def plot_word_frequencies_comparison(original_df, cleaned_df):
    """Plot word frequencies comparison before and after cleaning."""
    print("Plotting word frequency comparison...")
    
    # Compute word frequencies for the original data
    original_words = ' '.join(original_df[MESSAGE_COLUMN])
    original_tokens = word_tokenize(original_words)
    original_words = [word for word in original_tokens if word.isalpha() and word not in stop_words]
    original_word_freq = nltk.FreqDist(original_words)
    original_top_words = original_word_freq.most_common(20)
    
    # Compute word frequencies for the cleaned data
    cleaned_words = ' '.join(cleaned_df[NORMALIZED_MESSAGE_COLUMN])
    cleaned_tokens = word_tokenize(cleaned_words)
    cleaned_words = [word for word in cleaned_tokens if word.isalpha() and word not in stop_words]
    cleaned_word_freq = nltk.FreqDist(cleaned_words)
    cleaned_top_words = cleaned_word_freq.most_common(20)
    
    # Create DataFrames for plotting
    df_original = pd.DataFrame(original_top_words, columns=['Word', 'Frequency'])
    df_cleaned = pd.DataFrame(cleaned_top_words, columns=['Word', 'Frequency'])
    df_original['Dataset'] = 'Original'
    df_cleaned['Dataset'] = 'Cleaned'
    
    df_combined = pd.concat([df_original, df_cleaned], ignore_index=True)
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Frequency', y='Word', hue='Dataset', data=df_combined, palette='coolwarm')
    plt.title('Top 20 Words Frequency Comparison Before and After Cleaning')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()

def main():
    """Main function to execute the data cleaning and analysis pipeline."""
    # File paths
    input_file = r'C:\Users\Koon Kiat\OneDrive\Cloud\Phishing Detection\Datasets\emails.csv'
    output_file = 'cleaned_emails.csv'
    
    # Load dataset
    print("Loading dataset...")
    df_original = load_dataset(input_file)
    original_size = df_original.shape[0]
    print(f"Initial dataset size: {original_size} rows")

    # Data cleaning and normalization
    print("Removing duplicates...")
    df_no_duplicates = remove_duplicates(df_original)
    after_duplicates_size = df_no_duplicates.shape[0]
    print(f"Size after removing duplicates: {after_duplicates_size} rows")

    print("Handling missing values...")
    df_no_missing_values = handle_missing_values(df_no_duplicates)
    after_missing_values_size = df_no_missing_values.shape[0]
    print(f"Size after handling missing values: {after_missing_values_size} rows")

    print("Normalizing text data...")
    df_normalized = normalize_text_data(df_no_missing_values, num_cores=2)
    after_normalization_size = df_normalized.shape[0]
    print(f"Size after normalizing text data: {after_normalization_size} rows")

    # Extract required data
    print("Extracting data...")
    df_extracted = extract_data(df_normalized)
    
    # Visualization of cleaning process
    print("Visualizing the data cleaning process...")
    visualize_cleaning_process(
        original_size,
        after_duplicates_size,
        after_missing_values_size,
        after_normalization_size
    )
    
    # Compare with original data
    compare_data(df_original, df_normalized)
    
    # Plot word frequencies comparison
    plot_word_frequencies_comparison(df_original, df_normalized)
    
    # Save cleaned data
    print("Saving cleaned data to CSV file...")
    save_cleaned_data(df_extracted, output_file)
    
    print("Data cleaning and analysis completed successfully.")

if __name__ == "__main__":
    main()


# Define paths to email datasets
easy_ham_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\easy_ham\easy_ham"
hard_ham_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\hard_ham\hard_ham"
spam_path = r"C:\Users\Koon Kiat\OneDrive\Cloud\Projects\Phishing Email Detection\Spam Assassin\spam_2\spam_2"