import numpy as np  # Numerical operations
import pandas as pd 
import os
import logging
from SpamAndPhishingDetection import DatasetProcessor
from SpamAndPhishingDetection import log_label_percentages

 
logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ', level=logging.INFO)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(base_dir, 'Datasets', 'Phishing_Email.csv')
    df = pd.read_csv(dataset)  

    # Rename 'Email Type' column to 'Label' and map the values
    df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df = df.rename(columns={'Email Text': 'body'})

    # Drop the original 'Email Type' column if no longer needed
    df = df.drop(columns=['Email Type'])

    processor_phishing_emails = DatasetProcessor(df, "body", "Phishing Email")
    df_processed = processor_phishing_emails.process_dataset()

    log_label_percentages(df_processed, 'Phishing Email')



    # ************************* #
    #       Data Cleaning       #
    # ************************* #

   






if __name__ == '__main__':
    main()