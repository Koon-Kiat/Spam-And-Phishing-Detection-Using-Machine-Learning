import os
import pandas as pd
import logging
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from spellchecker import SpellChecker
import contractions


class TextProcessor:
    """
    A class for processing text data with various cleaning and preprocessing steps.

    Parameters
    ----------
    enable_spell_check : bool, optional
        Whether to enable spell checking. Default is False.

    Attributes
    ----------
    stop_words : set
        A set of stop words to be removed from the text.
    lemmatizer : WordNetLemmatizer
        An instance of WordNetLemmatizer for lemmatizing words.
    spell_checker : SpellChecker
        An instance of SpellChecker for spell checking.
    common_words : set
        A set of common words from the spell checker's word frequency.
    enable_spell_check : bool
        Whether spell checking is enabled.
    """

    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.common_words = set(self.spell_checker.word_frequency.keys())
        self.enable_spell_check = enable_spell_check
        logging.info("Initializing TextProcessor...")

    def expand_contractions(self, text):
        """
        Expand contractions in the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text with contractions expanded.
        """
        return contractions.fix(text)

    def remove_punctuation(self, text):
        """
        Remove punctuation from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without punctuation.
        """
        extra_punctuation = '“”‘’—–•·’'
        all_punctuation = string.punctuation + extra_punctuation
        return text.translate(str.maketrans('', '', all_punctuation))

    def tokenize(self, text):
        """
        Tokenize the text into words.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        list
            A list of words.
        """
        return word_tokenize(text)

    def remove_stop_words(self, words_list):
        """
        Remove stop words from the list of words.

        Parameters
        ----------
        words_list : list
            The list of words.

        Returns
        -------
        list
            The list of words without stop words.
        """
        return [w for w in words_list if w.lower() not in self.stop_words]

    def lemmatize(self, words_list):
        """
        Lemmatize the list of words.

        Parameters
        ----------
        words_list : list
            The list of words.

        Returns
        -------
        list
            The list of lemmatized words.
        """
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    def remove_urls(self, text):
        """
        Remove URLs from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without URLs.
        """
        return re.sub(r'(http[s]?|ftp):\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    def remove_custom_urls(self, text):
        """
        Remove custom URL patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without custom URL patterns.
        """
        return re.sub(r'\b(?:http|www)[^\s]*\b', '', text)

    def remove_numbers(self, text):
        """
        Remove numbers from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without numbers.
        """
        return re.sub(r'\d+', '', text)

    def remove_all_html_elements(self, text):
        """
        Remove all HTML elements from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without HTML elements.
        """
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {}
        return soup.get_text(separator=" ", strip=True)

    def remove_email_headers(self, text):
        """
        Remove email headers from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without email headers.
        """
        headers = ['From:', 'To:', 'Subject:', 'Cc:', 'Bcc:', 'Date:', 'Reply-To:', 'Content-Type:', 'Return-Path:', 'Message-ID:',
                   'Received:', 'MIME-Version:', 'Delivered-To:', 'Authentication-Results:', 'DKIM-Signature:', 'X-', 'Mail-To:']
        for header in headers:
            text = re.sub(rf'^{header}.*$', '', text, flags=re.MULTILINE)
        return text

    def remove_emails(self, text):
        """
        Remove email addresses from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without email addresses.
        """
        email_pattern_with_spaces = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_pattern_no_spaces = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        combined_pattern = f"({email_pattern_with_spaces}|{
            email_pattern_no_spaces})"
        return re.sub(combined_pattern, '', text)

    def remove_time(self, text):
        """
        Remove time patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without time patterns.
        """
        time_pattern = r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?: ?[APMapm]{2})?(?: [A-Z]{1,5})?\b'
        return re.sub(time_pattern, '', text)

    def remove_months(self, text):
        """
        Remove month names from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without month names.
        """
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        months_regex = r'\b(?:' + '|'.join(months) + r')\b'
        return re.sub(months_regex, '', text, flags=re.IGNORECASE)

    def remove_dates(self, text):
        """
        Remove date patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without date patterns.
        """
        date_pattern = (
            r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|'
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|'
            r'\b(?:\d{1,2}\s[A-Za-z]+\s\d{4})\b|'
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})\b'
        )
        return re.sub(date_pattern, '', text, flags=re.IGNORECASE)

    def remove_timezones(self, text):
        """
        Remove time zone patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without time zone patterns.
        """
        timezone_pattern = r'\b(?:[A-Z]{2,4}[+-]\d{2,4}|UTC|GMT|PST|EST|CST|MST)\b'
        return re.sub(timezone_pattern, '', text)

    def remove_multiple_newlines(self, text):
        """
        Remove multiple newlines from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text with multiple newlines replaced by a single newline.
        """
        return re.sub(r'\n{2,}', '\n', text)

    def remove_words(self, text):
        """
        Remove specific words from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without specific words.
        """
        return re.sub(r'\b(url|original message|submissionid|submission)\b', '', text, flags=re.IGNORECASE)

    def remove_single_characters(self, text):
        """
        Remove single characters from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without single characters.
        """
        return re.sub(r'\b\w\b', '', text)

    def remove_repetitive_patterns(self, text):
        """
        Remove repetitive patterns from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without repetitive patterns.
        """
        return re.sub(r'\b(nt+ts?|n+|t+|nt+)\b', '', text)

    def lowercase_text(self, text):
        """
        Convert the text to lowercase.

        Parameters:
        text (str): The input text.

        Returns:
        str: The text in lowercase.
        """
        return text.lower()

    def remove_bullet_points_and_symbols(self, text):
        """
        Remove bullet points and similar symbols from the text.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        str
            The text without bullet points and symbols.
        """
        symbols = ['•', '◦', '◉', '▪', '▫', '●', '□', '■',
                   '✦', '✧', '✪', '✫', '✬', '✭', '✮', '✯', '✰']
        for symbol in symbols:
            text = text.replace(symbol, '')
        return text

    def clean_text(self, X, y=None):
        """
        Clean and preprocess a list of text data.

        Parameters
        ----------
        X : list
            A list of text data to be cleaned.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the cleaned text data.
        """
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
        return pd.DataFrame({'cleaned_text': cleaned_text_list})

    def save_to_csv_cleaned(self, df, filename):
        """
        Save the cleaned text data to a CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the cleaned text data.
        filename : str
            The file path to save the CSV file.

        Returns
        -------
        None
        """
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")


def load_or_clean_data(dataset_name, df, text_column, file_path, cleaning_function):
    """
    Loads the data from the specified file path or cleans the data if the file does not exist.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being processed.
    df : pandas.DataFrame
        The DataFrame containing the data.
    text_column : str
        The name of the column containing text data to be cleaned.
    file_path : str
        The file path where the cleaned data will be saved.
    cleaning_function : function
        The function to clean the data.

    Returns
    -------
    pandas.DataFrame
        The loaded or cleaned DataFrame.
    """
    # logging.info(f"Loading or cleaning data...")
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading from file.")
        df_clean = pd.read_csv(file_path)
        df_clean['cleaned_text'] = df_clean['cleaned_text'].astype(
            str).fillna('')
        texts = df_clean['cleaned_text'].tolist()
        if not isinstance(texts, (list, tuple)):
            raise ValueError("Input should be a list or tuple of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError(
                "Input should be a list or tuple of strings. Found non-string elements.")

        return df_clean
    else:
        logging.info(f"File {file_path} does not exist. Cleaning data.")
        cleaned_df = cleaning_function(
            dataset_name, df, text_column, file_path)
        # logging.info(f"Data cleaning and saving to {file_path} completed.")

        return cleaned_df


def data_cleaning(dataset_name, df_processed, text_column, clean_file):
    """
    Cleans the text data in the specified column of the DataFrame.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being processed.
    df_processed : pandas.DataFrame
        The DataFrame containing the processed data.
    text_column : str
        The name of the column containing text data to be cleaned.
    clean_file : str
        The file path where the cleaned data will be saved.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    logging.info(f"Text processing {dataset_name} dataset...")
    processor = TextProcessor()
    df_clean = processor.clean_text(
        df_processed[text_column], df_processed['label'])
    processor.save_to_csv_cleaned(df_clean, clean_file)
    logging.info("Text processing and saving completed.")
    # logging.info(f"DataFrame columns after data cleaning: {df_clean.columns}")

    return df_clean


def combine_dataframes(combined_df, df_clean_body):
    logging.info(f"Combining Cleaned DataFrame with Merged DataFrame...")
    combined_df_reset = combined_df.reset_index(drop=True)
    df_clean_body_reset = df_clean_body.reset_index(drop=True)
    df_cleaned_combined = pd.concat([
        combined_df_reset[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
                           'short_urls', 'has_ip_address', 'urls', 'label']],
        df_clean_body_reset[['cleaned_text']]
    ], axis=1)
    logging.info(f"Dataframes combined successfully.\n")
    return df_cleaned_combined


def verify_combined_dataframe(combined_df, df_cleaned_combined):
    logging.info(f"Verifying the Cleaned Combined DataFrame...")
    combined_labels = combined_df['label'].unique()
    df_cleaned_combined_labels = df_cleaned_combined['label'].unique()
    if set(combined_labels) != set(df_cleaned_combined_labels):
        logging.error(f"Labels in Combined DataFrame do not match those in Cleaned Combined DataFrame. "
                      f"Combined DataFrame labels: {combined_labels}, "
                      f"Cleaned Combined DataFrame labels: {df_cleaned_combined_labels}")
        raise ValueError(
            "Labels do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info(
            "Labels in Combined DataFrame match those in Cleaned Combined DataFrame.")

    combined_label_counts = combined_df['label'].value_counts().sort_index()
    df_cleaned_combined_label_counts = df_cleaned_combined['label'].value_counts(
    ).sort_index()
    if not combined_label_counts.equals(df_cleaned_combined_label_counts):
        logging.error(
            "Label distributions in Combined DataFrame do not match those in Cleaned Combined DataFrame.")
        logging.error(f"Combined DataFrame distributions:\n{
                      combined_label_counts}")
        logging.error(f"Cleaned Combined DataFrame distributions:\n{
                      df_cleaned_combined_label_counts}")
        raise ValueError(
            "Label distributions do not match between Combined DataFrame and Cleaned Combined DataFrame.")
    else:
        logging.info(
            "Label distributions in Combined DataFrame match those in Cleaned Combined DataFrame.")


def save_dataframe_to_csv(df, file_path):
    df = df[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
             'short_urls', 'has_ip_address', 'urls', 'cleaned_text', 'label']]
    df.to_csv(file_path, index=False)
    logging.info(f"Data Cleaning completed.\n")


def combine_columns_for_cleaning(combined_df, df_clean_body):
    logging.info(f"Combining Cleaned DataFrame with Merged DataFrame...")
    combined_df_reset = combined_df.reset_index(drop=True)
    df_clean_body_reset = df_clean_body.reset_index(drop=True)
    df_cleaned_combined = pd.concat([
        combined_df_reset[['sender', 'receiver', 'https_count', 'http_count', 'blacklisted_keywords_count',
                           'short_urls', 'has_ip_address', 'urls', 'label']],
        df_clean_body_reset[['cleaned_text']]
    ], axis=1)
    logging.info(f"Dataframes combined successfully.\n")
    return df_cleaned_combined
