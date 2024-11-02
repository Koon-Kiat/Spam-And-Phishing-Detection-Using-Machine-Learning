import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BERTFeatureExtractor:
    """
    A class to extract BERT features from text data.

    Parameters
    ----------
    max_length : int, optional
        The maximum length of the tokenized sequences. Default is 128.
    device : torch.device, optional
        The device to run the BERT model on. Default is CUDA if available, otherwise CPU.

    Methods
    -------
    extract_features(texts, batch_size=16)
        Extracts BERT features from a list of text samples.
    """

    def __init__(self, max_length=128, device=None):
        logging.info("Initializing BERT Feature Extractor...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        # Determine the device
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Log the device being used
        logging.info(f"Using device: {self.device}")

        self.model.to(self.device)  # Ensure model is on the right device

    def extract_features(self, texts, batch_size=16):
        """
        Extract BERT features from a list of text samples.

        Parameters
        ----------
        texts : list
            A list of text samples.
        batch_size : int, optional
            The batch size for processing the text samples. Default is 16.

        Returns
        -------
        list
            A list of extracted BERT features.
        """
        if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input should be a list or tuple of strings.")

        features = []
        self.model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT features", leave=True):
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]
                tokens = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_features = outputs.last_hidden_state.mean(
                    dim=1).cpu().numpy()  # Move back to CPU
                features.extend(batch_features)

        return features


class BERTFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract BERT features from text data.

    Parameters
    ----------
    feature_extractor : BERTFeatureExtractor
        An instance of BERTFeatureExtractor to extract features.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data (no-op).
    transform(X)
        Transforms the data by extracting BERT features.
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, X, y=None):
        """
        Fit the transformer to the data (no-op).

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like, optional
            The target values (default is None).

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the data by extracting BERT features.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        numpy.ndarray
            The extracted BERT features.
        """
        return np.array(self.feature_extractor.extract_features(X))
