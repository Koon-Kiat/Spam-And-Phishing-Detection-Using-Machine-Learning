"""
Email headers cleaning utility functions.

This module provides functions for extracting and cleaning email headers,
focusing on extracting valid sender and receiver email addresses.
"""

import os
import re
import pandas as pd
import logging


def extract_email_address(raw_text):
    """
    Extract an email address from a string using regex patterns.

    Parameters
    ----------
    raw_text : str
        The text to parse for an email address.

    Returns
    -------
    str or None
        The extracted email address if valid, else None.
    """
    if isinstance(raw_text, str):
        # First try to extract email from angle brackets: "Name <email@domain.com>"
        match = re.search(r'<([^>]+)>', raw_text)
        if match:
            candidate = match.group(1).strip()
            if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', candidate):
                return candidate

        # If no match with angle brackets, check if the raw text itself is an email
        candidate = raw_text.strip()
        if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', candidate):
            return candidate
    return None
