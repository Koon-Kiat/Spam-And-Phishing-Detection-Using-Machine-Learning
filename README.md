# Detecting Spam and Phishing Emails Using Machine Learning

![GitHub repo size](https://img.shields.io/github/repo-size/Koon-Kiat/Phishing-Email-Detection)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning?style=flat)
![GitHub issues](https://img.shields.io/github/issues/Koon-Kiat/Phishing-Email-Detection)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning)
![GitHub Repo stars](https://img.shields.io/github/stars/Koon-Kiat/Detecting-Spam-and-Phishing-Emails-Using-Machine-Learning?style=flat)

## Overview

This project leverages advanced machine learning algorithms to detect and classify malicious emails, focusing on spam and phishing threats. As email threats grow more sophisticated, accurate detection is critical to ensuring the security and privacy of both individuals and organizations.

Our solution applies a combination of processes such as data preprocessing, feature engineering, and model training techniques to identify spam and phishing emails. The project addresses real-world challenges like imbalanced datasets by utilizing SpamAssassin and CEAS datasets for training and evaluation, ultimately enhancing the model's ability to filter phishing and spam emails effectively.

## Key Technologies

- **BERT for Feature Extraction**: Enhances contextual understanding of email content.
- **Stacked Ensemble Learning**: Combines XGBoost, Bagged SVM, and Logistic Regression for robust detection.
- **Optuna for Hyperparameter Tuning**: Optimizes model performance by fine-tuning key parameters.
- **Flask**: Provides a web interface for real-time email classification.

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

## Data

The project utilizes merged datasets from SpamAssassin (Hugging Face) and CEAS (Kaggle) to enhance email threat detection:

- **SpamAssassin**: Contains real-world spam and legitimate emails.
- **CEAS 2008**: Specially curated for anti-spam research, with a focus on phishing examples.

## Merging Datasets

TThe project integrates the **[Spam Assassin](https://huggingface.co/datasets/talby/spamassassin)** and **[CEAS 2008](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv)** datasets, aligning them by columns and ensuring label consistency. This creates a robust, well-labeled dataset that improves phishing and spam detection accuracy.



## Technology Stack

### Programming Languages

- **Python**

### Libraries and Frameworks

- **Machine Learning**: scikit-learn, TensorFlow, transformers, imbalanced-learn
- **NLP**: NLTK
- **Data Handling**: pandas, numpy
- **Web Framework**: Flask
- **Optimization**: Optuna

### Tools

- **Version Control**: Git
- **Environment Management**: Anaconda

### Additional Technologies

- **Feature Extraction**: BERT
- **Ensemble Learning**: XGBoost, Bagged SVM, Logistic Regression, Stacked Ensemble Learning
- **Data Preprocessing**: One-Hot Encoding, Standard Scaling, Imputation, Rare Category Removal, Noise Injection
- **Model Evaluation**: Stratified K-fold Cross-Validation, Confusion Matrix, Classification Reports
- **Regularization**: L1, L2
- **Noise Injection**: Adding controlled random variations to features to improve model generalization and reduce overfitting
- **Stacked Ensemble Learning**: Combining multiple models for robust detection

## Methodologies

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

- **Metrics**: Accuracy, precision, recall, F1-score.
- **Confusion Matrix**: Displaying the performance of each model.
- **Learning Curves**: Visualizing model performance as a function of training data size.

These results are stored in the `output` folder.

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

The accompanying Flask application provides a user-friendly interface where users can input email content for real-time spam and phishing detection. The system returns an analysis of whether an email is "Safe" or "Not Safe."

### Key Features:

- **User Interface**:

  - The main interface is provided by `index.html` and `taskpane.html` located in the `templates` folder.
  - Users can upload or paste email content for evaluation.

- **Instant Feedback**:

  - The `/evaluateEmail` endpoint processes the email content and returns immediate results, flagging malicious content.
  - This endpoint utilizes the `single_model_evaluation` module for classification.

- **Integration**:
  - The Flask app communicates with the machine learning model backend for classification.
  - Static assets such as icons are served from the `static/assets` folder.

### Example Usage:

To evaluate an email, users can navigate to the main interface, input the email content, and submit it for evaluation. The system will process the input and provide instant feedback on whether the email is "Safe" or "Not Safe."
