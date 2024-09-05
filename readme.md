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

## Usage
1. **Data Preprocessing**: Clean and preprocess the email data.
    ```python
    from text_processor import TextProcessor

    processor = TextProcessor(enable_spell_check=False)
    cleaned_text = processor.transform(raw_email_data)
    ```

2. **Feature Extraction**: Extract features from the cleaned text.
    ```python
    from feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features = extractor.transform(cleaned_text)
    ```

3. **Model Training**: Train a machine learning model.
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```

4. **Model Evaluation**: Evaluate the trained model.
    ```python
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ```

## Project Structure
Phishing-Detection/ │ ├── data/ # Raw and processed data ├── notebooks/ # Jupyter notebooks ├── src/ # Source code │ ├── data_preprocessing/ # Data preprocessing scripts │ ├── feature_engineering/ # Feature engineering scripts │ ├── models/ # Model training and evaluation scripts │ └── utils/ # Utility functions ├── tests/ # Unit tests ├── requirements.txt # Required Python packages ├── README.md # Project README └── .gitignore # Git ignore file

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [wordcloud](https://github.com/amueller/word_cloud)