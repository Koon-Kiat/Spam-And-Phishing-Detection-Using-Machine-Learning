# Phishing Detection Using Machine Learning

![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Email-Detection)
![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Email-Detection)
![Lines of code](https://img.shields.io/tokei/lines/github/Koon-Kiat/Phishing-Email-Detection)

## Overview
This project aims to detect phishing emails using machine learning techniques. It involves data preprocessing, feature engineering, model training, and evaluation. The goal is to build a robust classifier that can distinguish between phishing and legitimate emails.

## Features
- Data cleaning and preprocessing
- Feature engineering using BERT
- Model training and evaluation with various classifiers (e.g., Logistic Regression, Random Forest)
- Handling imbalanced data using SMOTE
- Visualization of results
- Progress bar for feature extraction
- Logging and warning management

## Installation Guide
1. Clone the repository:
    ```sh
    git clone https://github.com/Koon-Kiat/Phishing-Detection.git
    cd Phishing-Detection
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Download necessary NLTK resources:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage
1. **Data Cleaning and Preprocessing**:
    - Remove duplicates
    - Extract email information
    - Clean text data

2. **Feature Extraction**:
    - Use BERT for feature extraction
    - Handle imbalanced data using SMOTE

3. **Model Training and Evaluation**:
    - Train models using Logistic Regression, Random Forest, etc.
    - Evaluate models using accuracy, precision, recall, and F1 score

4. **Visualization**:
    - Plot word clouds
    - Visualize model performance

## Example
Here is an example of how to run the main processing function:

```python
from main import main

if __name__ == "__main__":
    main()