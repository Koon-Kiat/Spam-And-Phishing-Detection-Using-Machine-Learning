"""
Text processor module for cleaning and preprocessing text data.

This module provides a TextProcessor class that handles various text cleaning
and preprocessing steps for email content analysis.
"""

import logging
from tqdm import tqdm
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Import utility functions from utils module
from src.spamandphishingdetection.datacleaning.data_cleaning_utils import (
    expand_contractions, remove_punctuation, remove_urls, remove_custom_urls,
    remove_numbers, remove_all_html_elements, remove_email_headers,
    remove_email_addresses, remove_time_patterns, remove_month_names,
    remove_date_patterns, remove_timezone_references, remove_multiple_newlines,
    remove_specific_words, remove_single_characters, remove_repetitive_patterns,
    convert_to_lowercase, remove_bullet_points_and_symbols
)


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

    def clean_text(self, text_list, labels=None):
        """
        Clean and preprocess a list of text data.

        Parameters
        ----------
        text_list : list
            A list of text data to be cleaned.
        labels : pandas.Series, optional
            The labels corresponding to the text data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the cleaned text data.
        """
        cleaned_text_list = []
        non_string_count = 0
        nan_count = 0
        error_count = 0

        for body in tqdm(text_list, desc='Cleaning Text', unit='email'):
            try:
                # Convert non-string values to strings
                if not isinstance(body, str):
                    if pd.isna(body):
                        body = ""
                        nan_count += 1
                    else:
                        body = str(body)
                        non_string_count += 1

                # Apply all text cleaning functions
                text = remove_all_html_elements(body)
                text = expand_contractions(text)
                text = remove_email_headers(text)
                text = remove_email_addresses(text)
                text = remove_time_patterns(text)
                text = remove_month_names(text)
                text = remove_date_patterns(text)
                text = remove_timezone_references(text)
                text = remove_numbers(text)
                text = remove_multiple_newlines(text)
                text = remove_custom_urls(text)
                text = remove_urls(text)
                text = remove_punctuation(text)
                text = remove_specific_words(text)
                text = remove_single_characters(text)
                text = remove_repetitive_patterns(text)
                text = convert_to_lowercase(text)
                text = remove_bullet_points_and_symbols(text)

                # Process the text through tokenization, stop word removal, and lemmatization
                words_list = self.tokenize(text)
                words_list = self.remove_stop_words(words_list)
                lemmatized_list = self.lemmatize(words_list)
                cleaned_text_list.append(' '.join(lemmatized_list))
            except Exception as e:
                logging.error(f"Error processing text: {e}")
                error_count += 1
                cleaned_text_list.append('')

        # Log summary information instead of individual warnings
        if nan_count > 0:
            logging.warning(
                f"Found {nan_count} NaN values in text data, converted to empty strings")

        if non_string_count > 0:
            logging.warning(
                f"Found {non_string_count} non-string values in text data, converted to strings")

        if error_count > 0:
            logging.warning(
                f"Encountered {error_count} errors during text processing")

        return pd.DataFrame({'cleaned_text': cleaned_text_list})

    def save_cleaned_text(self, df, filename):
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
            logging.info(f"Cleaned data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving cleaned data to {filename}: {e}")
