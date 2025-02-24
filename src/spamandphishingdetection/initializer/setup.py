import os
import logging
import warnings
import nltk
import spacy
import tensorflow as tf
from transformers import logging as transformers_logging
from bs4 import MarkupResemblesLocatorWarning
from datetime import datetime
from pathlib import Path


BOLD = '\033[1m'
RESET = '\033[0m'


def initialize_environment(script_name):
    """Initializes the environment including logging configuration, NLTK resource downloads, and warning suppression.

    Args:
        script_name (str): Name of the script to set a base for the log file name.

    Returns:
        str: The path to the created log file.
    """
    log_folder = Path('logs')
    # Create log folder if it doesn't exist
    log_folder.mkdir(exist_ok=True)

    # Remove oldest log files if more than 5 exist
    log_files = list(log_folder.glob('*.log'))
    if len(log_files) > 5:
        log_files.sort(key=lambda f: f.stat().st_mtime)
        for f in log_files[:-5]:
            try:
                f.unlink()
            except Exception as e:
                logging.warning(f'Failed to remove log file {f}: {e}')

    base_script = Path(script_name).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{base_script}_{timestamp}.log"
    log_path = log_folder / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler()
        ]
    )

    try:
        # Download necessary NLTK resources quietly
        for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            nltk.download(resource, quiet=True)
    except Exception as e:
        logging.error(f'Error downloading NLTK resources: {e}')

    # Suppress TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('CRITICAL')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Suppress warnings globally and for specific modules
    warnings.simplefilter('ignore')
    warnings.filterwarnings(
        'ignore', category=UserWarning, module='transformers')
    warnings.filterwarnings('ignore', category=FutureWarning,
                            module='transformers.tokenization_utils_base')
    warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings(
        'ignore', category=UserWarning, module='tensorflow.keras')

    transformers_logging.set_verbosity_error()

    logging.info(f'Environment initialized with log file at {log_path}')
    return str(log_path)
