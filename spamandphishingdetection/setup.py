import os
import logging
import warnings
import nltk
import spacy
import tensorflow as tf
from transformers import logging as transformers_logging
from bs4 import MarkupResemblesLocatorWarning

# ANSI escape codes for text formatting
BOLD = '\033[1m'
RESET = '\033[0m'

def initialize_environment():
    # Download necessary NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nlp = spacy.load('en_core_web_sm')  # Load the spaCy English model

    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)

    # Suppress TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    # Set TensorFlow logger to suppress warnings
    tf.get_logger().setLevel('CRITICAL')
    # Configure the logging library to suppress TensorFlow logs
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Suppress warnings globally
    warnings.simplefilter("ignore")  # Ignore all warnings

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module='transformers.tokenization_utils_base')
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings("ignore", category=UserWarning,
                            module='tensorflow.keras')

    # Optionally, configure transformers logging
    transformers_logging.set_verbosity_error()

    # Define loss function using the recommended method
    loss_fn = tf.compat.v1.losses.sparse_softmax_cross_entropy

    return nlp, loss_fn