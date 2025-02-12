import os
import logging
import warnings
import nltk
import spacy
import tensorflow as tf
from transformers import logging as transformers_logging
from bs4 import MarkupResemblesLocatorWarning
from datetime import datetime

BOLD = '\033[1m'
RESET = '\033[0m'


def initialize_environment(script_name):
    # Create log folder if it doesn't exist
    log_folder = 'logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    base_script = os.path.basename(script_name).replace('.py', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{base_script}_{timestamp}.log"
    log_path = os.path.join(log_folder, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    # Download necessary NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Suppress TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Set TensorFlow logger to suppress warnings
    tf.get_logger().setLevel('CRITICAL')
    # Configure the logging library to suppress TensorFlow logs
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Suppress warnings globally
    warnings.simplefilter("ignore")

    # Suppress specific warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, module='transformers')
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module='transformers.tokenization_utils_base')
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings("ignore", category=UserWarning,
                            module='tensorflow.keras')

    # Configure transformers logging
    transformers_logging.set_verbosity_error()

    return log_path
