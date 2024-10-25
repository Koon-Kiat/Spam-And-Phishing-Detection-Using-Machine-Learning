# Standard Libraries
import re
import csv
import os
import logging
import string
from email import message_from_file
from email.utils import getaddresses
from typing import Dict, List
from collections import Counter
import warnings
import pickle
import shutil
import json

# Third-Party Libraries
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import contractions
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import numpy as np
import joblib

# Machine Learning Libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Data Augmentation
from imblearn.over_sampling import SMOTE

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*"
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ",
    level=logging.INFO,
)


class EmailHeaderExtractor:
    def count_https_http(self, text: str) -> Dict[str, int]:
        https_count = len(re.findall(r"https://", text))
        http_count = len(re.findall(r"http://", text))

        return {"https_count": https_count, "http_count": http_count}

    def contains_blacklisted_keywords(self, text: str) -> int:
        blacklisted_keywords = [
            "click now",
            "verify now",
            "urgent",
            "free",
            "winner",
            "limited time",
            "act now",
            "your account",
            "risk",
            "account update",
            "important update",
            "security alert",
            "confirm your identity",
            "password reset",
            "access your account",
            "log in",
            "claim your prize",
            "congratulations",
            "update required",
            "you have been selected",
            "validate your account",
            "final notice",
            "click here",
            "confirm now",
            "take action",
            "unauthorized activity",
            "sign in",
            "redeem now",
            "you are a winner",
            "download now",
            "urgent action required",
            "reset password",
            "limited offer",
            "exclusive deal",
            "verify account",
            "bank account",
            "payment declined",
            "upgrade required",
            "respond immediately",
        ]
        keyword_count = 0
        for keyword in blacklisted_keywords:
            keyword_count += len(re.findall(re.escape(keyword),
                                 text, re.IGNORECASE))

        return keyword_count

    def detect_url_shorteners(self, links: List[str]) -> int:
        shortener_domains = [
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "ow.ly",
            "t.co",
            "is.gd",
            "buff.ly",
            "adf.ly",
            "bl.ink",
            "lnkd.in",
            "shorte.st",
            "mcaf.ee",
            "q.gs",
            "po.st",
            "bc.vc",
            "s.coop",
            "u.to",
            "cutt.ly",
            "t2mio.com",
            "rb.gy",
            "clck.ru",
            "shorturl.at",
            "1url.com",
            "hyperurl.co",
            "urlzs.com",
            "v.gd",
            "x.co",
        ]
        short_urls = [
            link
            for link in links
            if any(domain in link for domain in shortener_domains)
        ]
        return len(short_urls)

    def count_ip_addresses(self, text: str) -> int:
        ip_pattern = r"https?://(\d{1,3}\.){3}\d{1,3}"

        return len(re.findall(ip_pattern, text))

    def count_urls(self, text: str) -> List[str]:
        url_pattern = r'https?:\/\/[^\s\'"()<>]+'
        links = re.findall(url_pattern, text)
        return links

    def extract_email_features_from_file(self, file_path):
        # Read the .eml file and parse the email
        with open(file_path, "r", encoding="utf-8") as f:
            msg = message_from_file(f)

        # Extract sender and receiver
        sender = ", ".join([addr for name, addr in getaddresses(
            [msg.get("From")])]) if msg.get("From") else "unknown"
        receiver = ", ".join([addr for name, addr in getaddresses(
            [msg.get("To")])]) if msg.get("To") else "unknown"

        # Extract body (assuming plain text)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(
                        decode=True).decode(charset, "ignore")
                    break
        else:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, "ignore")

        # Count occurrences of https and http
        https_http_counts = self.count_https_http(body)
        blacklisted_keyword_count = self.contains_blacklisted_keywords(body)
        urls = self.count_urls(body)
        short_urls = self.detect_url_shorteners(urls)
        has_ip_address = self.count_ip_addresses(body)
        url_count = len(urls)

        # Return extracted features as a dictionary
        return {
            "sender": sender,
            "receiver": receiver,
            "https_count": https_http_counts["https_count"],
            "http_count": https_http_counts["http_count"],
            "blacklisted_keywords_count": blacklisted_keyword_count,
            "urls": url_count,
            "short_urls": short_urls,
            "has_ip_address": has_ip_address,
            "url_count": url_count,
            "body": body,
        }

    def save_to_csv(self, email_features, csv_file_path):
        # Save the extracted features to a CSV file
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = email_features.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header
            writer.writerow(email_features)


def extract_email(text):
    if isinstance(text, str):
        match = re.search(r"<([^>]+)>", text)
        if match:
            return match.group(1)
        elif re.match(r"^[^@]+@[^@]+\.[^@]+$", text):
            return text
    return None


class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, enable_spell_check=False):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.common_words = set(self.spell_checker.word_frequency.keys())
        self.enable_spell_check = enable_spell_check
        logging.info("Initializing TextProcessor...")

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_punctuation(self, text):
        extra_punctuation = "“”‘’—–•·’"
        all_punctuation = string.punctuation + extra_punctuation
        return text.translate(str.maketrans("", "", all_punctuation))

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, words_list):
        return [w for w in words_list if w.lower() not in self.stop_words]

    def lemmatize(self, words_list):
        return [self.lemmatizer.lemmatize(w) for w in words_list]

    def remove_urls(self, text):
        return re.sub(
            r"(http[s]?|ftp):\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
            # Updated to remove non-standard URL formats (e.g., 'http ww newsisfree com')
        )

    def remove_custom_urls(self, text):
        return re.sub(
            r"\b(?:http|www)[^\s]*\b", "", text
        )  # Catch patterns like 'http ww' or 'www.' that are incomplete

    def remove_numbers(self, text):
        return re.sub(r"\d+", "", text)

    def remove_all_html_elements(self, text):
        soup = BeautifulSoup(text, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {}

        return soup.get_text(separator=" ", strip=True)

    def remove_email_headers(self, text):
        headers = [
            "From:",
            "To:",
            "Subject:",
            "Cc:",
            "Bcc:",
            "Date:",
            "Reply-To:",
            "Content-Type:",
            "Return-Path:",
            "Message-ID:",
            "Received:",
            "MIME-Version:",
            "Delivered-To:",
            "Authentication-Results:",
            "DKIM-Signature:",
            "X-",
            "Mail-To:",
        ]
        for header in headers:
            text = re.sub(rf"^{header}.*$", "", text, flags=re.MULTILINE)

        return text

    def remove_emails(self, text):
        # Regex pattern to match emails with or without spaces around "@"
        email_pattern_with_spaces = r"\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        # Regex pattern to match emails without spaces
        email_pattern_no_spaces = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        # Combine both patterns using the OR operator
        combined_pattern = f"({email_pattern_with_spaces}|{
            email_pattern_no_spaces})"

        return re.sub(combined_pattern, "", text)

    def remove_time(self, text):
        # Regex to match various time patterns
        time_pattern = r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?: ?[APMapm]{2})?(?: [A-Z]{1,5})?\b"

        return re.sub(time_pattern, "", text)

    def remove_months(self, text):
        months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "jan",
            "feb",
            "mar",
            "apr",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
        months_regex = r"\b(?:" + "|".join(months) + r")\b"

        return re.sub(months_regex, "", text, flags=re.IGNORECASE)

    def remove_dates(self, text):
        date_pattern = (
            # Example: Mon, 2 Sep 2002
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|"
            # Example: 20-09-2002, Sep 13 2002
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]+\s\d{1,2},\s\d{4})\b|"
            r"\b(?:\d{1,2}\s[A-Za-z]+\s\d{4})\b|"  # Example: 01 September 2002
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"  # Example: 24/08/2002
        )

        return re.sub(date_pattern, "", text, flags=re.IGNORECASE)

    def remove_timezones(self, text):
        # Regex to match time zones (e.g., PST, EST, GMT, UTC)
        timezone_pattern = r"\b(?:[A-Z]{2,4}[+-]\d{2,4}|UTC|GMT|PST|EST|CST|MST)\b"

        return re.sub(timezone_pattern, "", text)

    def remove_multiple_newlines(self, text):
        return re.sub(
            r"\n{2,}", "\n", text
        )  # Replace multiple newlines with a single newline

    def remove_words(self, text):
        return re.sub(
            r"\b(url|original message|submissionid|submission)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )  # Combine all words using the | (OR) operator in regex

    def remove_single_characters(self, text):
        return re.sub(
            r"\b\w\b", "", text
        )  # Remove single characters that are not part of a word

    def remove_repetitive_patterns(self, text):
        return re.sub(
            r"\b(nt+ts?|n+|t+|nt+)\b", "", text
        )  # Combine patterns for 'nt+ts?', repetitive 'n' or 'nt', and 't+', 'n+', 'nt+'

    def lowercase_text(self, text):
        return text.lower()

    def remove_bullet_points_and_symbols(self, text):
        symbols = [
            "•",
            "◦",
            "◉",
            "▪",
            "▫",
            "●",
            "□",
            "■",
            "✦",
            "✧",
            "✪",
            "✫",
            "✬",
            "✭",
            "✮",
            "✯",
            "✰",
        ]  # List of bullet points and similar symbols
        for symbol in symbols:
            text = text.replace(symbol, "")

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_text_list = []

        for body in tqdm(X, desc="Cleaning Text", unit="email"):
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
                cleaned_text_list.append(" ".join(lemmatized_list))
            except Exception as e:
                logging.error(f"Error processing text: {e}")
                cleaned_text_list.append("")
        if y is not None:
            logging.info(f"Total amount of text processed: {
                         len(cleaned_text_list)}")

            return pd.DataFrame({"cleaned_text": cleaned_text_list, "label": y})
        else:
            logging.info(f"Total amount of text processed: {
                         len(cleaned_text_list)}")

            return pd.DataFrame({"cleaned_text": cleaned_text_list})

    def save_to_csv_cleaned(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")


class BERTFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(self.feature_extractor.extract_features(X))


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
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Ensure that the dimensions are correctly handled
        input_ids = inputs["input_ids"].squeeze(dim=0)
        attention_mask = inputs["attention_mask"].squeeze(dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTFeatureExtractor:
    def __init__(self, max_length=128, device=None):
        logging.info("Initializing BERT Feature Extractor...")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", clean_up_tokenization_spaces=True
        )
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)  # Ensure model is on the right device

    def extract_features(self, texts, batch_size=16):
        if not isinstance(texts, (list, tuple)) or not all(
            isinstance(text, str) for text in texts
        ):
            raise ValueError("Input should be a list or tuple of strings.")
        features = []
        self.model.eval()
        with torch.no_grad():
            for start in tqdm(
                range(0, len(texts), batch_size),
                desc="Extracting BERT features",
                leave=True,
            ):
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_features = (
                    outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                )  # Move back to CPU
                features.extend(batch_features)

        return features


def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    base_dir = config["base_dir"]
    TestEmail = os.path.join(base_dir, "SingleEvaluation", "TestEmail.eml")
    SavedEmail = os.path.join(base_dir, "SingleEvaluation", "FormatedTest.csv")
    CleanedEmail = os.path.join(
        base_dir, "SingleEvaluation", "CleanedTest.csv")
    MergedEmail = os.path.join(base_dir, "SingleEvaluation", "MergedTest.csv")
    SavedModel = os.path.join(
        base_dir, "SingleEvaluation", "Random Forest.pkl"
    )
    # source_path = os.path.join(
    #     base_dir, "Models & Parameters", "ensemble_model_fold_1.pkl"
    # )
    # shutil.copy(source_path, SavedModel)

    # Extract features from the email
    extractor = EmailHeaderExtractor()
    features = extractor.extract_email_features_from_file(TestEmail)
    extractor.save_to_csv(features, SavedEmail)

    # Clean the column ['sender'] and ['receiver']
    df = pd.read_csv(SavedEmail, encoding="utf-8")
    df["sender"] = df["sender"].apply(extract_email)
    df["receiver"] = df["receiver"].apply(extract_email)
    df.to_csv(SavedEmail, index=False, encoding="utf-8")

    # Clean the text
    processor = TextProcessor()
    df = pd.read_csv(SavedEmail, encoding="utf-8")
    df_clean = processor.transform(df["body"])
    processor.save_to_csv_cleaned(df_clean, CleanedEmail)

    # Combine cleaned text with the dataframe
    df_clean = pd.read_csv(CleanedEmail)
    df = pd.read_csv(SavedEmail)
    df_combined = pd.concat([df, df_clean], axis=1)
    df_combined.drop(columns=["body"], inplace=True)
    df_combined.to_csv(MergedEmail, index=False)

    # Feature Extraction
    categorical_columns = ["sender", "receiver"]
    numerical_columns = [
        "https_count",
        "http_count",
        "blacklisted_keywords_count",
        "urls",
        "short_urls",
        "has_ip_address",
        "url_count",
    ]
    text_column = "cleaned_text"

    bert_extractor = BERTFeatureExtractor()
    bert_transformer = BERTFeatureTransformer(feature_extractor=bert_extractor)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        (
                            "encoder",
                            OneHotEncoder(sparse_output=False,
                                          handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_columns,
            ),
        ],
        # Keep other columns unchanged, like 'cleaned_text' and 'label'
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("bert_features", bert_transformer),  # Custom transformer for BERT
            ("pca", PCA(n_components=10)),
        ]
    )

    # Load the cleaned email data
    # Load the cleaned email data
    email_df = pd.read_csv(MergedEmail)

    # Fit the non-text features
    pipeline.named_steps["preprocessor"].fit(
        email_df.drop(columns="cleaned_text"))

    # Transform the non-text features
    email_df_non_text_transformed = pipeline.named_steps["preprocessor"].transform(
        email_df
    )

    # Transform the text features
    texts = email_df["cleaned_text"].tolist()
    email_df_text_processed = pipeline.named_steps["bert_features"].transform(
        texts)

    # Combine the features
    email_df_combined = np.hstack(
        (email_df_non_text_transformed, email_df_text_processed)
    )

    # Apply PCA only if more than one sample
    if email_df_combined.shape[0] > 1:
        pca = PCA(n_components=10)
        email_pca = pca.fit_transform(email_df_combined)
    else:
        # Set email_pca to an empty array with shape (1, 10) to match model expectations
        email_pca = np.zeros((1, 777))  # Placeholder for a single prediction

    # Load the pre-trained model
    model = joblib.load(SavedModel)

    # Make predictions on the PCA-transformed or placeholder data
    predictions = model.predict(email_pca)

    # Map predictions to labels
    label_map = {0: "safe", 1: "not safe"}
    for pred in predictions:
        print(f"The email is {label_map[pred]}.")
        return f"The email is {label_map[pred]}."


if __name__ == "__main__":
    main()
