# Detecting Spam and Phishing Emails Using Machine Learning

![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Email-Detection)
![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Email-Detection)

## Overview

This project leverages advanced machine learning algorithms to detect and classify malicious emails, focusing on spam and phishing threats. As email threats grow more sophisticated, accurate detection is critical to ensuring the security and privacy of both individuals and organizations.

Our solution applies a combination of processes such as data preprocessing, feature engineering, and model training techniques to identify spam and phishing emails. While we implemented specific steps for data cleaning and feature extraction, these were not integrated into a single machine learning pipeline. The project addresses real-world challenges like imbalanced datasets by utilizing SpamAssassin and CEAS datasets for training and evaluation, ultimately enhancing the model's ability to filter phishing and spam emails effectively.

Key Technologies:

- **BERT for Feature Extraction**: We use Bidirectional Encoder Representations from Transformers (BERT) to enhance contextual understanding of email content.
- **Stacked Ensemble Learning**: The model ensemble combines XGBoost, Bagged SVM, and Logistic Regression, providing a robust solution for detecting phishing emails.
- **Optuna for Hyperparameter Tuning**: Optuna optimizes the model's performance by fine-tuning key parameters.

The project also includes a Flask web application that serves as a user interface, enabling users to receive real-time classifications on their inboxes on whether emails are "Safe" or "Not Safe."

## Installation

To set up the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/phishing-email-detection.git
cd phishing-email-detection
pip install -r requirements.txt
```

Once the dependencies are installed, you can run the phishing email detection program using the following command:

```bash
python SpamAndPhishingDetection.py
```

## Data

The project utilizes merged datasets from SpamAssassin (Hugging Face) and CEAS (Kaggle) to enhance email threat detection:

- **SpamAssassin**: Contains real-world spam and legitimate emails.
- **CEAS 2008**: Specially curated for anti-spam research, with a focus on phishing examples.

## Merging Datasets

TThe project integrates the **[Spam Assassin](https://huggingface.co/datasets/talby/spamassassin)** and **[CEAS 2008](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv)** datasets, aligning them by columns and ensuring label consistency. This creates a robust, well-labeled dataset that improves phishing and spam detection accuracy.

### File Structure for Storing Results

- Data Cleaning/
- Data Integration/
- Data Preprocessing/
- Noise Injection/
- Data Splitting/
- Feature Engineering/
- Feature Extraction/
- Models & Parameters/

## Model Training Methodologies

The following methodologies were employed to enhance the model's performance:

- **BERT Feature Extraction**: Leveraging BERT to extract contextual features from email content, which improves the model's understanding of the text.

- **Ensemble Learning**: Combining multiple models (e.g., Logistic Regression, SVM, XGBoost) using meta-modeling techniques to improve predictive performance and robustness.'

- **Noise Injection**: Adding controlled random variations to features to improve model generalization and reduce overfitting.

- **Hyperparameter Optimization with Optuna**: Using Optuna to systematically tune model hyperparameters, ensuring optimal settings for better accuracy and reduced overfitting.

- **Cross-Validation**: Implementing stratified K-fold cross-validation to ensure the model generalizes well across different subsets of data.

- **Preprocessing Pipelines**: Utilizing pipelines that include:

  - **Rare Category Removal**: To eliminate categories with very few occurrences, enhancing model performance.
  - **Imputation**: Filling missing values in the dataset.
  - **Encoding**: Applying One-Hot Encoding for categorical variables.
  - **Standard Scaling**: Normalizing numerical features for improved model performance.

## Evaluation

- **Accuracy**: Measures overall prediction correctness.
- **Precision, Recall, F1-Score:** Evaluates the balance between correct and incorrect classifications.
- **Confusion Matrix**: Displays the performance of each model in predicting "Safe" vs. "Not Safe" emails.
- **Learning Curve**: A plot showing model performance (accuracy/loss) as a function of training data size, helping to visualize overfitting, underfitting, and the effectiveness of adding more training data.

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

## Flask Application

The accompanying Flask application provides a user-friendly interface where users can input email content for real-time phishing detection. The system returns an analysis of whether an email is "Safe" or "Not Safe."

### Key Features:

- **User Interface**: A simple email input form that allows users to upload or paste email content.
- **Instant Feedback**: Provides immediate results, flagging malicious content.
- **Integration**: The app communicates with the machine learning model backend for classification.

### Future Enhancements

The project will continue evolving with the goal of improving model scalability and enhancing integration with platforms like Microsoft Outlook for automatic phishing email detection and flagging.

## Additional Files

- **Dataset Evaluation**: Utilized for assessing the model on an additional dataset.
- **Multi-Model Evaluation**: Employed for batch testing models on sample emails.
