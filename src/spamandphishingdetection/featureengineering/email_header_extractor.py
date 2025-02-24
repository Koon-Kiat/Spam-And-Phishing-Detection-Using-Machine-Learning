import logging
import os
import pandas as pd
import re
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Dict, List, Union
from tqdm import tqdm


class EmailHeaderExtractor:
    """
    A class to extract various email header features for spam detection from email datasets.

    Attributes:
        df (pd.DataFrame): DataFrame containing email text or body to extract from.
        headers_df (pd.DataFrame): DataFrame storing the extracted header features.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EmailHeaderExtractor with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the emails.
        """
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")

    def clean_links(self, links: List[str]) -> List[str]:
        """
        Clean a list of URL strings by removing unwanted characters.

        Args:
            links (List[str]): List of URL strings.

        Returns:
            List[str]: List of cleaned URL strings.
        """
        cleaned_links = []
        for link in links:
            link = re.sub(r'[\'\[\]\s]+', '', link)
            link = re.sub(r'\\n+', ' ', link)
            link = link.strip()
            if link:
                cleaned_links.append(link)
        return cleaned_links

    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        """
        Extract inline email headers (From, To, Mail-To) from raw email text using regex.

        Args:
            email_text (str): Raw email text.

        Returns:
            Dict[str, Union[str, None]]: Dictionary with keys 'From', 'To', and 'Mail-To'.
        """
        from_match = re.search(
            r'From:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        to_match = re.search(
            r'To:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        mail_to_match = re.search(
            r'mailto:.*?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
        from_header = from_match.group(1) if from_match else None
        to_header = to_match.group(1) if to_match else None
        mail_to_header = mail_to_match.group(1) if mail_to_match else None
        return {'From': from_header, 'To': to_header, 'Mail-To': mail_to_header}

    def extract_body_content(self, email_message: EmailMessage) -> str:
        """
        Extract the body content from an EmailMessage object, handling both plain text and HTML parts.

        Args:
            email_message (EmailMessage): Parsed email message.

        Returns:
            str: Combined email body content.
        """
        body_content = ""
        if email_message.is_multipart():
            for part in email_message.iter_parts():
                if part.get_content_type() in ['text/plain', 'text/html']:
                    body_content += part.get_payload(
                        decode=True).decode(errors='ignore')
        else:
            body_content = email_message.get_payload(
                decode=True).decode(errors='ignore')
        return body_content

    def count_https_http(self, text: str) -> Dict[str, int]:
        """
        Count occurrences of 'http://' and 'https://' in the given text.

        Args:
            text (str): Text to search.

        Returns:
            Dict[str, int]: Dictionary with counts for 'https_count' and 'http_count'.
        """
        https_count = len(re.findall(r'https://', text))
        http_count = len(re.findall(r'http://', text))
        return {'https_count': https_count, 'http_count': http_count}

    def count_blacklisted_keywords(self, text: str) -> int:
        """
        Count occurrences of pre-defined blacklisted keywords that are common in spam messages.

        Args:
            text (str): Text to search.

        Returns:
            int: Total count of blacklisted keyword occurrences.
        """
        blacklisted_keywords = [
            'click now', 'verify now', 'urgent', 'free', 'winner',
            'limited time', 'act now', 'your account', 'risk', 'account update',
            'important update', 'security alert', 'confirm your identity',
            'password reset', 'access your account', 'log in', 'claim your prize',
            'congratulations', 'update required', 'you have been selected',
            'validate your account', 'final notice', 'click here', 'confirm now',
            'take action', 'unauthorized activity', 'sign in', 'redeem now',
            'you are a winner', 'download now', 'urgent action required',
            'reset password', 'limited offer', 'exclusive deal', 'verify account',
            'bank account', 'payment declined', 'upgrade required', 'respond immediately'
        ]
        keyword_count = 0
        for keyword in blacklisted_keywords:
            keyword_count += len(re.findall(re.escape(keyword),
                                 text, re.IGNORECASE))
        return keyword_count

    def detect_url_shorteners(self, links: List[str]) -> int:
        """
        Detect URL shorteners in a list of URLs based on known shortener domains.

        Args:
            links (List[str]): List of URL strings.

        Returns:
            int: Count of URLs that are shortened using known shortener domains.
        """
        shortener_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly',
            'adf.ly', 'bl.ink', 'lnkd.in', 'shorte.st', 'mcaf.ee', 'q.gs', 'po.st',
            'bc.vc', 's.coop', 'u.to', 'cutt.ly', 't2mio.com', 'rb.gy', 'clck.ru',
            'shorturl.at', '1url.com', 'hyperurl.co', 'urlzs.com', 'v.gd', 'x.co'
        ]
        short_urls = [link for link in links if any(
            domain in link for domain in shortener_domains)]
        return len(short_urls)

    def count_ip_addresses(self, text: str) -> int:
        """
        Count occurrences of IP addresses in the given text using a regex pattern.

        Args:
            text (str): Text to search.

        Returns:
            int: Count of IP addresses found.
        """
        ip_pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'
        return len(re.findall(ip_pattern, text))

    def extract_headers_spamassassin(self) -> pd.DataFrame:
        """
        Extract email header features from the SpamAssassin dataset.
        Processes the raw email text to retrieve headers and body features.

        Returns:
            pd.DataFrame: DataFrame containing extracted features.
        """
        headers_list: List[Dict[str, Union[str, List[str], int]]] = []
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                email_message = BytesParser(policy=policy.default).parsebytes(
                    email_text.encode('utf-8'))
                subject = email_message['Subject'] if email_message['Subject'] else None
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get(
                    'Mail-To') if email_message.get('Mail-To') else None

                if not from_header or not to_header:
                    inline_headers = self.extract_inline_headers(email_text)
                    from_header = inline_headers['From'] or from_header
                    to_header = inline_headers['To'] or to_header
                    mail_to_header = inline_headers['Mail-To'] or mail_to_header

                body_content = self.extract_body_content(email_message)
                logging.debug(f"Email body content: {body_content}")

                # Additional robust feature extraction
                exclamation_count = body_content.count('!')
                uppercase_count = len(re.findall(
                    r'\b[A-Z]{2,}\b', body_content))
                body_length = len(body_content)
                special_chars_count = len(re.findall(r'[^\w\s]', body_content))

                # Extract URLs from body content
                url_pattern = r'https?:\/\/[^\s\'"()<>]+'
                links = re.findall(url_pattern, body_content)
                links = self.clean_links(links)

                # Count occurrences
                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.count_blacklisted_keywords(
                    body_content)
                short_urls = self.detect_url_shorteners(links)
                has_ip_address = self.count_ip_addresses(body_content)

                headers_list.append({
                    'sender': from_header,
                    'receiver': to_header,
                    'mailto': mail_to_header,
                    'subject': subject,
                    'texturls': links,
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address,
                    'exclamation_count': exclamation_count,
                    'uppercase_count': uppercase_count,
                    'body_length': body_length,
                    'special_chars_count': special_chars_count
                })
            except Exception as e:
                logging.error(f"Error parsing email: {e}")
                headers_list.append({
                    'sender': None,
                    'receiver': None,
                    'mailto': None,
                    'subject': None,
                    'texturls': [],
                    'https_count': 0,
                    'http_count': 0,
                    'blacklisted_keywords_count': 0,
                    'short_urls': 0,
                    'has_ip_address': 0,
                    'exclamation_count': 0,
                    'uppercase_count': 0,
                    'body_length': 0,
                    'special_chars_count': 0
                })
        self.headers_df = pd.DataFrame(headers_list)
        return self.headers_df

    def extract_headers_ceas(self) -> pd.DataFrame:
        """
        Extract email header features from the CEAS_08 dataset.
        Assumes that the DataFrame contains email bodies directly in the 'body' column.

        Returns:
            pd.DataFrame: DataFrame containing extracted features.
        """
        headers_list: List[Dict[str, int]] = []
        for email_text in tqdm(self.df['body'], desc="Extracting headers"):
            try:
                body_content = email_text
                logging.debug(f"Email body content: {body_content}")

                # Additional robust feature extraction
                exclamation_count = body_content.count('!')
                uppercase_count = len(re.findall(
                    r'\b[A-Z]{2,}\b', body_content))
                body_length = len(body_content)
                special_chars_count = len(re.findall(r'[^\w\s]', body_content))

                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.count_blacklisted_keywords(
                    body_content)
                links = self.clean_links(re.findall(
                    r'https?:\/\/[^\s\'"()<>]+', body_content))
                short_urls = self.detect_url_shorteners(links)
                has_ip_address = self.count_ip_addresses(body_content)

                headers_list.append({
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address,
                    'exclamation_count': exclamation_count,
                    'uppercase_count': uppercase_count,
                    'body_length': body_length,
                    'special_chars_count': special_chars_count
                })
            except Exception as e:
                logging.error(f"Error processing email: {e}")
                headers_list.append({
                    'https_count': 0,
                    'http_count': 0,
                    'blacklisted_keywords_count': 0,
                    'short_urls': 0,
                    'has_ip_address': 0,
                    'exclamation_count': 0,
                    'uppercase_count': 0,
                    'body_length': 0,
                    'special_chars_count': 0
                })
        self.headers_df = pd.DataFrame(headers_list)
        return self.headers_df

    def save_to_csv(self, file_path: str):
        """
        Save the extracted header features to a CSV file.

        Args:
            file_path (str): Path where the CSV file will be saved.

        Raises:
            ValueError: If no header information has been extracted.
        """
        if not self.headers_df.empty:
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError(
                "No header information extracted. Please run extract_headers() first.")


def load_or_extract_headers(df: pd.DataFrame, file_path: str, extractor_class, dataset_type: str) -> pd.DataFrame:
    """
    Load pre-extracted email headers from a CSV file or extract them using the provided extractor class.

    Args:
        df (pd.DataFrame): DataFrame with email data.
        file_path (str): Path to the CSV file for storing header data.
        extractor_class: Class for extracting headers.
        dataset_type (str): Type of dataset ('Spam Assassin' or 'CEAS_08').

    Returns:
        pd.DataFrame: DataFrame with email header features.

    Raises:
        ValueError: If an unknown dataset type is provided.
    """
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading from file.")
        return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist.")
        logging.info(f"Extracting headers for dataset: {dataset_type}.")
        header_extractor = extractor_class(df)
        if dataset_type == "Spam Assassin":
            headers_df = header_extractor.extract_headers_spamassassin()
        elif dataset_type == "CEAS_08":
            headers_df = header_extractor.extract_headers_ceas()
        else:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Please specify either 'Spam Assassin' or 'CEAS_08'.")
        header_extractor.save_to_csv(file_path)
        logging.info(
            f"Email header extraction and saving to {file_path} completed for dataset: {dataset_type}.\n")
        return headers_df


def count_urls(urls_list: list) -> int:
    """
    Count the number of URLs in a list.

    Args:
        urls_list (list): List containing URLs.

    Returns:
        int: Count of URLs if list; otherwise 0.
    """
    return len(urls_list) if isinstance(urls_list, list) else 0


def process_spamassassin_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process SpamAssassin header DataFrame by counting URLs and dropping unnecessary columns.

    Args:
        df (pd.DataFrame): DataFrame with spamassassin header features.

    Returns:
        pd.DataFrame: Processed DataFrame with updated features.
    """
    df['urls'] = df['texturls'].apply(count_urls)
    df.drop(columns=['mailto', 'texturls'], inplace=True)
    return df


def feature_engineering(df_processed_spamassassin: pd.DataFrame, df_processed_ceas: pd.DataFrame, file_paths: dict) -> tuple:
    """
    Perform feature engineering by extracting and processing email header features from both SpamAssassin and CEAS datasets.

    Args:
        df_processed_spamassassin (pd.DataFrame): DataFrame with SpamAssassin raw email data.
        df_processed_ceas (pd.DataFrame): DataFrame with CEAS_08 raw email bodies.
        file_paths (dict): Dictionary containing file paths for saving extracted headers.

    Returns:
        tuple: A tuple containing processed SpamAssassin and CEAS header DataFrames.
    """

    spamassassin_headers_df = load_or_extract_headers(
        df_processed_spamassassin,
        file_paths['extracted_spam_assassin_email_header_file'],
        EmailHeaderExtractor,
        'Spam Assassin'
    )
    spamassassin_headers_df = process_spamassassin_headers(
        spamassassin_headers_df)

    ceas_headers_df = load_or_extract_headers(
        df_processed_ceas,
        file_paths['extracted_ceas_email_header_file'],
        EmailHeaderExtractor,
        'CEAS_08'
    )

    return spamassassin_headers_df, ceas_headers_df
