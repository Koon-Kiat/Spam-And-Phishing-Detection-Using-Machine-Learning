import os
import json
import logging
import joblib
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from optuna.samplers import TPESampler


def load_or_save_model(model, model_path, action='load'):
    """
    Loads or saves the model based on the specified action.

    Parameters
    ----------
    model : object
        The model to be loaded or saved.
    model_path : str
        The file path where the model will be saved or loaded from.
    action : str, optional
        The action to perform ('load' or 'save'). Default is 'load'.

    Returns
    -------
    object
        The loaded model if action is 'load'.
    """
    if action == 'load':
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
        else:
            logging.info(
                f"No saved model found at {model_path}. Proceeding to train a new model.")
            return None
    elif action == 'save':
        logging.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)


def load_or_save_params(params, params_path, action='load'):
    """
    Loads or saves the parameters based on the specified action.

    Parameters
    ----------
    params : dict
        The parameters to be loaded or saved.
    params_path : str
        The file path where the parameters will be saved or loaded from.
    action : str, optional
        The action to perform ('load' or 'save'). Default is 'load'.

    Returns
    -------
    dict
        The loaded parameters if action is 'load'.
    """
    if action == 'load':
        if os.path.exists(params_path):
            logging.info(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f)
        else:
            logging.info(
                f"No saved parameters found at {params_path}. Proceeding to conduct a study.")
            return None
    elif action == 'save':
        logging.info(f"Saving parameters to {params_path}")
        with open(params_path, 'w') as f:
            json.dump(params, f)


def model_training(X_train, y_train, X_test, y_test, model_path, params_path):
    """
    Trains the model using the provided training data and evaluates it on the test data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.
    X_test : pandas.DataFrame
        The test data features.
    y_test : pandas.Series
        The test data labels.
    model_path : str
        The file path where the model will be saved.
    params_path : str
        The file path where the parameters will be saved.

    Returns
    -------
    tuple
        The trained ensemble model and the test accuracy.
    """
    try:
        ensemble_model = load_or_save_model(None, model_path, action='load')
        best_params = load_or_save_params(None, params_path, action='load')
    except Exception as e:
        logging.error(f"Error loading model or parameters: {e}")
        ensemble_model = None
        best_params = None

    # Train a new model if no existing model or parameters are found
    if ensemble_model is None and best_params is None:
        logging.info(
            "No existing ensemble model or parameters found. Conducting Optuna study and training model...")
        best_params = conduct_optuna_study(X_train, y_train)
        load_or_save_params(best_params, params_path, action='save')
        ensemble_model = train_ensemble_model(
            best_params, X_train, y_train, model_path)
    elif ensemble_model is None and best_params is not None:
        logging.info(
            "Parameters found, but no ensemble model. Training new model with existing parameters...")
        ensemble_model = train_ensemble_model(
            best_params, X_train, y_train, model_path)
    elif ensemble_model is not None and best_params is None:
        logging.info(
            "Ensemble model found, but no parameters. Using pre-trained model for evaluation.")
    else:
        logging.info(
            "Ensemble model and parameters found. Using pre-trained model for evaluation.")

    # Make predictions
    y_train_pred = ensemble_model.predict(X_train)
    y_test_pred = ensemble_model.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    target_names = ['Safe', 'Not Safe']

    # Print the performance metrics
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

    return ensemble_model, test_accuracy


def conduct_optuna_study(X_train, y_train):
    """
    Conducts an Optuna study to find the best hyperparameters for the models.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.

    Returns
    -------
    dict
        The best hyperparameters for each model.
    """
    best_params = {}

    # Optimize XGBoost parameters

    def xgb_objective(trial):
        n_estimators_xgb = trial.suggest_int('n_estimators_xgb', 50, 100)
        max_depth_xgb = trial.suggest_int('max_depth_xgb', 3, 10)
        learning_rate_xgb = trial.suggest_float('learning_rate_xgb', 0.01, 0.3)
        # Increase the range for stronger L1 regularization
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
        # Increase the range for stronger L2 regularization
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

        model = XGBClassifier(
            n_estimators=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=learning_rate_xgb,
            reg_alpha=reg_alpha,  # L1
            reg_lambda=reg_lambda,  # L2
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        return accuracy_score(y_train, y_train_pred)

    # Optimize XGBoost parameters
    xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    xgb_study.optimize(xgb_objective, n_trials=5)
    best_params['xgb'] = xgb_study.best_params

    def rf_objective(trial):
        try:
            # Define hyperparameters for the RandomForest model
            n_estimators_rf = trial.suggest_int(
                'n_estimators_rf', 50, 300)  # Number of trees
            max_depth_rf = trial.suggest_int(
                'max_depth_rf', 3, 20)  # Maximum depth of the trees
            min_samples_split_rf = trial.suggest_int(
                'min_samples_split_rf', 2, 10)  # Minimum samples to split a node
            # Minimum samples to be at a leaf node
            min_samples_leaf_rf = trial.suggest_int(
                'min_samples_leaf_rf', 1, 4)

            # Create RandomForest model with the suggested hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators_rf,
                max_depth=max_depth_rf,
                min_samples_split=min_samples_split_rf,
                min_samples_leaf=min_samples_leaf_rf,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            # Fit the model and evaluate performance
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)

            # Return the training accuracy score
            return accuracy_score(y_train, y_train_pred)
        except Exception as e:
            logging.error(f"Error in Random Forest objective function: {e}")
            return 0  # Return a low score if there's an error


# Optimize RandomForest parameters using Optuna
    rf_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    rf_study.optimize(rf_objective, n_trials=5)

    # Store the best parameters for RandomForest in best_params dictionary
    best_params['rf'] = rf_study.best_params

    def logreg_objective(trial):
        try:
            # Set a lower upper bound for stronger regularization
            C_logreg = trial.suggest_float('C_logreg', 0.0001, 1.0)
            penalty = trial.suggest_categorical(
                'penalty', ['l1', 'l2'])  # Regularization type

            model = LogisticRegression(
                C=C_logreg,
                penalty=penalty,
                solver='saga' if penalty == 'l1' else 'lbfgs',  # saga for L1, lbfgs for L2
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)
        except Exception as e:
            logging.error(
                f"Error in Logistic Regression objective function: {e}")
            return 0  # Return a low score if there's an error

    # Optimize Logistic Regression parameters
    logreg_study = optuna.create_study(
        direction='maximize', sampler=TPESampler())
    logreg_study.optimize(logreg_objective, n_trials=5)
    best_params['logreg'] = logreg_study.best_params

    return best_params


def load_optuna_model(path):
    """
    Loads the Optuna model from the specified path.

    Parameters
    ----------
    path : str
        The file path where the model is saved.

    Returns
    -------
    object
        The loaded Optuna model.
    """
    return joblib.load(path)
