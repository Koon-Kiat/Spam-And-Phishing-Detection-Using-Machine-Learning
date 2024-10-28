import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open(os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')) as config_file:
    config = json.load(config_file)
    base_dir = config['base_dir']

# Define model_path
model_path = os.path.join(base_dir, 'additional_model_training', 'base_model')


def model_training(X_train, y_train, X_test, y_test, model, model_name):
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Safe', 'Not Safe']

    # Print the performance metrics
    print(f"Model: {model_name}")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(
        f"Classification Report for Training Data:\n{classification_report(y_train, y_train_pred, target_names=target_names)}")
    print(
        f"\nClassification Report for Test Data:\n{classification_report(y_test, y_test_pred, target_names=target_names)}")
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)

    # Add labels, title, and other aesthetics
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Data')
    plt.show()

    joblib.dump(model, os.path.join(model_path, f"{model_name}.pkl"))
    return model, test_accuracy
