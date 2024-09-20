# Phishing Detection Using Machine Learning

![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Email-Detection)
![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Email-Detection)
![Lines of code](https://img.shields.io/tokei/lines/github/Koon-Kiat/Phishing-Email-Detection)

## Overview
The Phishing Email Detection project leverages advanced machine learning techniques to identify and classify phishing emails. By employing a comprehensive pipeline that includes data preprocessing, feature engineering, and model training, the project aims to build a highly accurate and robust classifier capable of distinguishing between legitimate emails and potential phishing threats.

This project utilizes state-of-the-art techniques such as BERT for feature extraction, ensemble learning methods including Logistic Regression, Random Forest, SVM, and XGBoost, and advanced hyperparameter tuning through Optuna. Additionally, it addresses challenges like imbalanced datasets using SMOTE and incorporates effective preprocessing strategies such as rare category removal and feature scaling. The goal is to enhance email security by providing reliable classification to help users identify phishing attempts.

## Installation

To set up the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/phishing-email-detection.git
cd phishing-email-detection
pip install -r requirements.txt
```

## Data
The dataset consists of labeled emails, which are preprocessed and stored in the following structure:
- Data Cleaning/
- Data Integration/
- Data Preprocessing/
- Data Splitting/
- Feature Engineering/
- Feature Extraction/
- Models & Parameters/

## Merging Datasets
The project supports merging multiple datasets to create a more robust training set. Specifically, it combines the **[Spam Assassin](https://huggingface.co/datasets/talby/spamassassin)** dataset from Hugging Face and the **[CEAS_08](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv)** dataset from Kaggle. This approach enhances model performance by providing a richer set of features and examples, leading to more reliable phishing detection.


## Model Training Methodologies
The following methodologies were employed to enhance the model's performance:

- **BERT Feature Extraction:** Leveraging BERT to extract contextual features from email content, which improves the model's understanding of the text.

- **Ensemble Learning:** Combining multiple models (e.g., Logistic Regression, SVM, XGBoost) using meta-modeling techniques to improve predictive performance and robustness.

- **Hyperparameter Optimization with Optuna:** Using Optuna to systematically tune model hyperparameters, ensuring optimal settings for better accuracy and reduced overfitting.

- **Cross-Validation:** Implementing stratified K-fold cross-validation to ensure the model generalizes well across different subsets of data.

- **Preprocessing Pipelines:** Utilizing pipelines that include:

    - Rare Category Removal: To eliminate categories with very few occurrences, enhancing model performance.
    - Imputation: Filling missing values in the dataset.
    - Encoding: Applying One-Hot Encoding for categorical variables.
    - Standard Scaling: Normalizing numerical features for improved model performance.

## Evaluation
The model's performance is evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- Classification Report
- Learning Plot Curve


### Example Output
```
Training Accuracy: XX.XX%
Test Accuracy: XX.XX%
Confusion Matrix:
[[TN FP]
 [FN TP]]
Classification Report for Test Data:
              precision    recall  f1-score   support
       Safe       0.XX      0.XX      0.XX      XX
   Not Safe       0.XX      0.XX      0.XX      XX
```
