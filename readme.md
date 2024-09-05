# Phishing Detection Using Machine Learning

![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Detection)
![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Detection)
![Lines of code](https://img.shields.io/tokei/lines/github/Koon-Kiat/Phishing-Detection)

## Overview
This project aims to detect phishing emails using machine learning techniques. It involves data preprocessing, feature engineering, model training, and evaluation. The goal is to build a robust classifier that can distinguish between phishing and legitimate emails.

## Features
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Visualization of results

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

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [wordcloud](https://github.com/amueller/word_cloud)