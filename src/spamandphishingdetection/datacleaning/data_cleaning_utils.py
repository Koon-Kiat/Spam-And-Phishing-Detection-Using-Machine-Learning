"""
Utility functions for text processing in the spam and phishing detection project.

This module provides text cleaning and preprocessing utility functions that are used
by the data cleaning pipeline.
"""

import re
import string
import logging
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import contractions


def expand_contractions(text):
    """
    Expand contractions in the text (e.g., "don't" -> "do not").

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


def remove_punctuation(text):
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
    extra_punctuation = '""''—–•·'  # Fixed the unterminated string literal
    all_punctuation = string.punctuation + extra_punctuation
    return text.translate(str.maketrans('', '', all_punctuation))


def remove_urls(text):
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


def remove_custom_urls(text):
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


def remove_numbers(text):
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


def remove_all_html_elements(text):
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


def remove_email_headers(text):
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


def remove_email_addresses(text):
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
    combined_pattern = f"({email_pattern_with_spaces}|{email_pattern_no_spaces})"
    return re.sub(combined_pattern, '', text)


def remove_time_patterns(text):
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


def remove_month_names(text):
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


def remove_date_patterns(text):
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


def remove_timezone_references(text):
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


def remove_multiple_newlines(text):
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


def remove_specific_words(text):
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


def remove_single_characters(text):
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


def remove_repetitive_patterns(text):
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


def convert_to_lowercase(text):
    """
    Convert the text to lowercase.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text in lowercase.
    """
    return text.lower()


def remove_bullet_points_and_symbols(text):
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
