"""
Data Cleaning Package
-------------------
This package provides tools and utilities for cleaning email data.

The package includes functionality for cleaning both email body text
and email headers, with support for batch processing and pipeline operations.
"""

from src.spamandphishingdetection.datacleaning.data_cleaning_pipeline_api import (
    process_text_cleaning,
    run_data_cleaning_pipeline,
    clean_email_text
)

from src.spamandphishingdetection.datacleaning.headers_api import (
    run_headers_cleaning_pipeline,
    clean_single_email_header
)

__all__ = [
    'process_text_cleaning',
    'run_data_cleaning_pipeline',
    'clean_email_text',
    'run_headers_cleaning_pipeline',
    'clean_single_email_header'
]
