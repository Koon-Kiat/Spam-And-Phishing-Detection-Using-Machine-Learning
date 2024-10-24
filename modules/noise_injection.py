import os
import random
import logging
import pandas as pd
import numpy as np

def generate_noisy_dataframe(data, file_path, noise_level=0.1):
    """
    Generates a noisy DataFrame by injecting noise into specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The original DataFrame to be processed.
    file_path : str
        The path to save or load the noisy DataFrame.
    noise_level : float
        The level of noise to inject (probability for categorical/text, standard deviation for numerical).

    Returns
    -------
    pandas.DataFrame
        The noisy DataFrame.
    """

    # Function to inject noise into numerical columns
    def inject_numerical_noise(data, columns, noise_level):
        for column in columns:
            noise = np.random.normal(0, noise_level, data[column].shape)
            data[column] += noise
        return data

    # Function to inject noise into text columns
    def inject_text_noise(data, text_column, noise_level):
        for i in range(len(data)):
            if random.random() < noise_level:  # With a probability of noise_level
                text_list = list(data[text_column][i])
                if len(text_list) > 0:  # Ensure text_list is not empty
                    # Choose a random position
                    pos = random.randint(0, len(text_list) - 1)
                    # Replace with a random letter
                    text_list[pos] = random.choice(
                        'abcdefghijklmnopqrstuvwxyz')
                    data.at[i, text_column] = ''.join(text_list)
        return data

    # Function to inject noise into categorical columns
    def inject_categorical_noise(data, columns, noise_level):
        for column in columns:
            unique_values = data[column].unique()
            for i in range(len(data)):
                if random.random() < noise_level:  # With a probability of noise_level
                    data.at[i, column] = random.choice(unique_values)
        return data

    # Check if the noisy DataFrame already exists
    if os.path.exists(file_path):
        logging.info(f"Noisy DataFrame already exists as '{
                     file_path}'. Loading it.")
        df_noisy = pd.read_csv(file_path)
    else:
        logging.info(
            f"'{file_path}' does not exist. Generating a noisy DataFrame.")

        # Define the columns for noise injection
        numerical_columns = ['https_count', 'http_count',
                             'blacklisted_keywords_count', 'urls', 'short_urls', 'has_ip_address']
        categorical_columns = ['sender', 'receiver']

        # Apply noise injection
        data = inject_numerical_noise(data, numerical_columns, noise_level)
        data = inject_text_noise(data, 'cleaned_text', noise_level)
        data = inject_categorical_noise(data, categorical_columns, noise_level)

        # Save the noisy DataFrame
        data.to_csv(file_path, index=False)
        logging.info(f"Noisy DataFrame saved as '{file_path}'.")

    return data
