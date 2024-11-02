import os

from .file_operations import load_config, get_file_paths
from .feature_engineering import EmailHeaderExtractor, load_or_extract_headers
from .pipeline import run_pipeline_or_load

__version__ = "0.1.0"