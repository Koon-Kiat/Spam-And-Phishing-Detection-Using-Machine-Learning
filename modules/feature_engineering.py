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
    A class to extract email headers and other relevant information from email data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the email data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.headers_df = pd.DataFrame()
        logging.info("Initializing EmailHeaderExtractor...")

    def clean_links(self, links: List[str]) -> List[str]:
        """
        Clean the extracted links by removing unwanted characters and spaces.

        Parameters
        ----------
        links : List[str]
            The list of links to be cleaned.

        Returns
        -------
        List[str]
            The cleaned list of links.
        """
        cleaned_links = []
        for link in links:
            link = re.sub(r'[\'\[\]\s]+', '', link)
            link = re.sub(r'\\n+', ' ', link)
            link = link.strip()  # Trim leading and trailing spaces
            if link:  # Avoid appending empty links
                cleaned_links.append(link)

        return cleaned_links

    def extract_inline_headers(self, email_text: str) -> Dict[str, Union[str, None]]:
        """
        Extract inline headers (From, To, Mail-To) from the email text.

        Parameters
        ----------
        email_text : str
            The email text to extract headers from.

        Returns
        -------
        Dict[str, Union[str, None]]
            A dictionary containing the extracted headers.
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
        Extract the body content from an email message.

        Parameters
        ----------
        email_message : EmailMessage
            The email message to extract the body content from.

        Returns
        -------
        str
            The extracted body content.
        """
        body_content = ""
        if email_message.is_multipart():
            for part in email_message.iter_parts():
                if part.get_content_type() == 'text/plain':
                    body_content += part.get_payload(
                        decode=True).decode(errors='ignore')
                elif part.get_content_type() == 'text/html':
                    body_content += part.get_payload(
                        decode=True).decode(errors='ignore')
        else:
            body_content = email_message.get_payload(
                decode=True).decode(errors='ignore')

        return body_content

    def count_https_http(self, text: str) -> Dict[str, int]:
        """
        Count the occurrences of 'https' and 'http' in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        Dict[str, int]
            A dictionary containing the counts of 'https' and 'http'.
        """
        https_count = len(re.findall(r'https://', text))
        http_count = len(re.findall(r'http://', text))

        return {'https_count': https_count, 'http_count': http_count}

    def contains_blacklisted_keywords(self, text: str) -> int:
        """
        Count the occurrences of blacklisted keywords in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        int
            The count of blacklisted keywords in the text.
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
        Detect the number of URL shorteners in the list of links.

        Parameters
        ----------
        links : List[str]
            The list of links to check for URL shorteners.

        Returns
        -------
        int
            The count of URL shorteners in the list of links.
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
        Count the occurrences of IP addresses in the text.

        Parameters
        ----------
        text : str
            The text to count the occurrences in.

        Returns
        -------
        int
            The count of IP addresses in the text.
        """
        ip_pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'

        return len(re.findall(ip_pattern, text))

    def extract_headers_spamassassin(self) -> pd.DataFrame:
        """
        Extract headers and other relevant information from the email data for SpamAssassin dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted headers and information.
        """
        headers_list: List[Dict[str, Union[str, List[str], int]]] = []
        for email_text in tqdm(self.df['text'], desc="Extracting headers"):
            try:
                email_message = BytesParser(policy=policy.default).parsebytes(
                    email_text.encode('utf-8'))
                from_header = email_message['From'] if 'From' in email_message else None
                to_header = email_message['To'] if 'To' in email_message else None
                mail_to_header = email_message.get(
                    'Mail-To') if email_message.get('Mail-To') else None

                if not from_header or not to_header:
                    inline_headers = self.extract_inline_headers(email_text)
                    from_header = inline_headers['From'] or from_header
                    to_header = inline_headers['To'] or to_header
                    mail_to_header = inline_headers['Mail-To'] or mail_to_header

                from_header = from_header if from_header else None
                to_header = to_header if to_header else None
                mail_to_header = mail_to_header if mail_to_header else None
                body_content = self.extract_body_content(email_message)
                logging.debug(f"Email body content: {body_content}")

                # Extract URLs from body content
                url_pattern = r'https?:\/\/[^\s\'"()<>]+'
                links = re.findall(url_pattern, body_content)
                links = self.clean_links(links)

                # Count blacklisted keywords, http/https, short URLs, and IP addresses in the email body
                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.contains_blacklisted_keywords(
                    body_content)
                short_urls = self.detect_url_shorteners(links)
                has_ip_address = self.count_ip_addresses(body_content)

                headers_list.append({
                    'sender': from_header,
                    'receiver': to_header,
                    'mailto': mail_to_header,
                    'texturls': links,
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address
                })
            except Exception as e:
                logging.error(f"Error parsing email: {e}")
                headers_list.append(
                    {'sender': None, 'receiver': None, 'mailto': None, 'texturls': [], 'blacklisted_keywords_count': 0, 'short_urls': [], 'has_ip_address': 0})
        self.headers_df = pd.DataFrame(headers_list)

        return self.headers_df

    def extract_headers_ceas(self) -> pd.DataFrame:
        """
        Extract headers and other relevant information from the email data for CEAS dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the extracted headers and information.
        """
        headers_list: List[Dict[str, int]] = []

        for email_text in tqdm(self.df['body'], desc="Extracting headers"):
            try:
                body_content = email_text  # Assuming 'email_text' contains the email body directly
                logging.debug(f"Email body content: {body_content}")

                # Count blacklisted keywords and http/https occurrences in the email body
                https_http_counts = self.count_https_http(body_content)
                blacklisted_keyword_count = self.contains_blacklisted_keywords(
                    body_content)
                short_urls = self.detect_url_shorteners(self.clean_links(
                    re.findall(r'https?:\/\/[^\s\'"()<>]+', body_content)))
                has_ip_address = self.count_ip_addresses(body_content)

                headers_list.append({
                    'https_count': https_http_counts['https_count'],
                    'http_count': https_http_counts['http_count'],
                    'blacklisted_keywords_count': blacklisted_keyword_count,
                    'short_urls': short_urls,
                    'has_ip_address': has_ip_address
                })
            except Exception as e:
                logging.error(f"Error processing email: {e}")
                headers_list.append({
                    'https_count': 0,
                    'http_count': 0,
                    'blacklisted_keywords_count': 0,
                    'short_urls': [],
                    'has_ip_address': 0
                })
        self.headers_df = pd.DataFrame(headers_list)

        return self.headers_df

    def save_to_csv(self, file_path: str):
        """
        Save the extracted headers DataFrame to a CSV file.

        Parameters
        ----------
        file_path : str
            The path to save the CSV file.

        Raises
        ------
        ValueError
            If no header information has been extracted.
        """
        if not self.headers_df.empty:
            self.headers_df.to_csv(file_path, index=False)
            logging.info(f"Data successfully saved to: {file_path}")
        else:
            raise ValueError(
                "No header information extracted. Please run extract_headers() first.")


def load_or_extract_headers(df: pd.DataFrame, file_path: str, extractor_class, dataset_type: str) -> pd.DataFrame:
    """
    Loads the email headers from the specified file path or extracts them if the file does not exist.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    file_path : str
        The file path where the extracted headers will be saved.
    extractor_class : class
        The class used to extract the headers.
    dataset_type : str
        The type of dataset being processed.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with extracted headers.
    """
    logging.info("Loading or extracting email headers...")
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists. Loading from file.")

        return pd.read_csv(file_path)
    else:
        logging.info(f"File {file_path} does not exist. Extracting headers for dataset: {
                     dataset_type}.")
        header_extractor = extractor_class(df)

        # Check dataset type and call the corresponding extraction function
        if dataset_type == "Spam Assassin":
            headers_df = header_extractor.extract_headers_spamassassin()
        elif dataset_type == "CEAS_08":
            headers_df = header_extractor.extract_headers_ceas()
        else:
            raise ValueError(f"Unknown dataset type: {
                             dataset_type}. Please specify either 'Spam Assassin' or 'CEAS_08'.")
        header_extractor.save_to_csv(file_path)
        logging.info(f"Email header extraction and saving to {
                     file_path} completed for dataset: {dataset_type}.")

        return headers_df


def count_urls(urls_list):
    """
    Counts the number of URLs in the provided list.

    Parameters
    ----------
    urls_list : list
        The list of URLs.

    Returns
    -------
    int
        The number of URLs in the list.
    """
    if isinstance(urls_list, list):
        return len(urls_list)
    else:
        return 0


def process_spamassassin_headers(df):
    df['urls'] = df['texturls'].apply(count_urls)
    df.drop(columns=['mailto', 'texturls'], inplace=True)
    return df

def feature_engineering(df_processed_spamassassin, df_processed_ceas, file_paths):
    logging.info(f"Beginning Feature Engineering...")

    # Extract email headers from the SpamAssassin dataset
    spamassassin_headers_df = load_or_extract_headers(
        df_processed_spamassassin, file_paths['extracted_spam_assassin_email_header_file'], EmailHeaderExtractor, 'Spam Assassin')
    spamassassin_headers_df = process_spamassassin_headers(spamassassin_headers_df)

    ceas_headers_df = load_or_extract_headers(
        df_processed_ceas, file_paths['extracted_ceas_email_header_file'], EmailHeaderExtractor, 'CEAS_08')

    logging.info(f"Feature Engineering completed.\n")
    return spamassassin_headers_df, ceas_headers_df