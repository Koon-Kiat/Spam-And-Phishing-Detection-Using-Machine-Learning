# Spam & Phishing Detection Using Machine Learning

![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Email-Detection)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning?style=flat)
![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Email-Detection)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning)
![GitHub Repo stars](https://img.shields.io/github/stars/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning?style=flat)

## Overview

This project leverages advanced machine learning algorithms to detect and classify malicious emails, focusing on spam and phishing threats. As email threats grow more sophisticated, accurate detection is critical to ensuring the security and privacy of both individuals and organizations.

Our solution applies a combination of processes such as data preprocessing, feature engineering, and model training techniques to identify spam and phishing emails. The project addresses real-world challenges like imbalanced datasets by utilizing SpamAssassin and CEAS datasets for training and evaluation, ultimately enhancing the model's ability to filter phishing and spam emails effectively.

## Key Technologies

- **Programming Language**: Python
- **ML/DL Libraries**: scikit-learn, TensorFlow, transformers, imbalanced-learn
- **NLP**: NLTK
- **Data Processing**: pandas, numpy
- **Development Tools**: Git, Anaconda
- **Optimization**: Optuna
- **Feature Extraction**: BERT
- **Ensemble Learning**: XGBoost, Bagged SVM, Logistic Regression, Stacked Ensemble Learning
- **Data Preprocessing**: One-Hot Encoding, Standard Scaling, Imputation, Rare Category Removal, Noise Injection
- **Model Evaluation**: Stratified K-fold Cross-Validation, Confusion Matrix, Classification Reports
- **Regularization**: L1, L2
- **Noise Injection**: Adding controlled random variations to features to improve model generalization and reduce overfitting
- **Stacked Ensemble Learning**: Combining multiple models for robust detection

## Features

- **Advanced Spam and Phishing Detection**: Utilizes sophisticated algorithms to accurately identify malicious emails.
- **Support for Handling Imbalanced Datasets**: Implements techniques to manage and balance skewed data distributions.
- **Automated Model Training and Evaluation**: Streamlines the process of training and assessing machine learning models.

## Methodologies

### Data Sources

The project utilizes merged datasets from SpamAssassin (Hugging Face) and CEAS (Kaggle) to enhance email threat detection:

- **SpamAssassin**: Contains real-world spam and legitimate emails.
- **CEAS 2008**: Specially curated for anti-spam research, with a focus on phishing examples.

### Data Preprocessing

- **Cleaning**: Removing duplicates, handling missing values, and correcting errors.
- **Imputation**: Filling in missing values using appropriate strategies.
- **Scaling**: Normalizing or standardizing features to improve model performance.

### Feature Engineering

- **BERT for Feature Extraction**: Enhancing contextual understanding of email content.
- **Other Techniques**: Applying additional feature extraction methods to improve model accuracy.

### Data Integration

- **Merging Datasets**: Combining SpamAssassin and CEAS datasets.
- **Ensuring Consistency**: Aligning columns and labels for a unified dataset.

### Noise Injection

- **Controlled Variations**: Adding random variations to features to improve model generalization and reduce overfitting.

### Cross-Validation

- **Stratified K-fold**: Ensuring model generalization by maintaining the proportion of classes in each fold.

### Model Training

- **Ensemble Learning**: Using techniques like XGBoost, Bagged SVM, and Logistic Regression for robust detection.

### Evaluation

- **Accuracy**: Measures overall prediction correctness.
- **Precision, Recall, F1-Score:** Evaluates the balance between correct and incorrect classifications.
- **Confusion Matrix**: Displays the performance of each model in predicting "Safe" vs. "Not Safe" emails.
- **Learning Curve**: A plot showing model performance (accuracy/loss) as a function of training data size, helping to visualize overfitting, underfitting, and the effectiveness of adding more training data.


## Installation

To set up the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning
cd Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning
conda create --name <your_environment_name> python=3.8.20
conda activate <your_environment_name>
conda env update --file environment.yaml --prune
```

Once the dependencies are installed, you can run the phishing email detection program using the following command:

```bash
python main.py
```

### Example Output

```
Training Accuracy: XX.XX%
Test Accuracy:    XX.XX%

Confusion Matrix:
[[ TN  FP ]
 [ FN  TP ]]
Classification Report for Training Data:
              precision    recall  f1-score   support
       Safe       0.XX      0.XX      0.XX      XX
   Not Safe       0.XX      0.XX      0.XX      XX

   accuracy                           0.XX      XX
  macro avg       0.XX      0.XX      0.XX      XX
weighted avg      0.XX      0.XX      0.XX      XX


Classification Report for Test Data:
              precision    recall  f1-score   support
       Safe       0.XX      0.XX      0.XX      XX
   Not Safe       0.XX      0.XX      0.XX      XX

   accuracy                           0.XX      XX
  macro avg       0.XX      0.XX      0.XX      XX
weighted avg      0.XX      0.XX      0.XX      XX
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SpamAssassin Public Corpus
- CEAS 2008 Dataset Contributors
- Open Source ML Community
