import os
import json
import logging
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna  # Hyperparameter optimization
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tqdm import tqdm


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
        f"Classification Report for Training Data: \n{classification_report(y_train, y_train_pred, target_names=target_names)}")
    print(
        f"\nClassification Report for Test Data: \n{classification_report(y_test, y_test_pred, target_names=target_names)}")

    return ensemble_model, test_accuracy


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

    def svm_objective(trial):
        try:
            # Regularization parameter for SVM
            C_svm = trial.suggest_float('C_svm', 0.1, 1.0)
            kernel_svm = trial.suggest_categorical(
                'kernel_svm', ['linear', 'rbf', 'poly'])

            model = SVC(
                C=C_svm,
                kernel=kernel_svm,
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)
        except Exception as e:
            logging.error(f"Error in SVM objective function: {e}")
            return 0  # Return a low score if there's an error

    # Optimize SVM parameters
    svm_study = optuna.create_study(direction='maximize', sampler=TPESampler())
    svm_study.optimize(svm_objective, n_trials=5)
    best_params['svm'] = svm_study.best_params

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


def train_ensemble_model(best_params, X_train, y_train, model_path):
    """
    Trains an ensemble model using the best hyperparameters.

    This function trains a stacking ensemble model consisting of a Bagged SVM and an XGBoost model as base models,
    with a Logistic Regression model as the meta-model. The best hyperparameters for each model are provided
    through the `best_params` dictionary. The trained model is saved to the specified file path.

    Args:
        best_params (dict): The best hyperparameters for each model.
            - 'xgb': Hyperparameters for the XGBoost model.
                - 'n_estimators_xgb' (int): Number of boosting rounds.
                - 'max_depth_xgb' (int): Maximum tree depth for base learners.
                - 'learning_rate_xgb' (float): Boosting learning rate.
                - 'reg_alpha' (float, optional): L1 regularization term on weights (default is 0.0).
                - 'reg_lambda' (float, optional): L2 regularization term on weights (default is 1.0).
            - 'svm': Hyperparameters for the SVM model.
                - 'C_svm' (float): Regularization parameter.
                - 'kernel_svm' (str): Specifies the kernel type to be used in the algorithm.
            - 'logreg': Hyperparameters for the Logistic Regression model.
                - 'C_logreg' (float): Inverse of regularization strength (smaller values specify stronger regularization).
                - 'penalty' (str, optional): Used to specify the norm used in the penalization ('l1' or 'l2', default is 'l2').
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels.
        model_path (str): The file path where the model will be saved.

    Returns:
        object: The trained ensemble model.
    """
    logging.info(f"Training new ensemble model with best parameters")

    # XGBoost model with increased L1 and L2 regularization
    xgb_model = XGBClassifier(
        n_estimators=best_params['xgb']['n_estimators_xgb'],
        max_depth=best_params['xgb']['max_depth_xgb'],
        learning_rate=best_params['xgb']['learning_rate_xgb'],
        # L1 regularization (default 0.0)
        reg_alpha=best_params['xgb'].get('reg_alpha', 0.0),
        # L2 regularization (default 1.0)
        reg_lambda=best_params['xgb'].get('reg_lambda', 1.0),
        # Adjust for class imbalance
        scale_pos_weight=len(y_train[y_train == 0]) / \
        len(y_train[y_train == 1]),
        random_state=42,
        n_jobs=2
    )

    # Bagged SVM Model
    bagged_svm = BaggingClassifier(
        estimator=SVC(
            # Regularization strength for SVM (higher C = less regularization)
            C=best_params['svm']['C_svm'],
            kernel=best_params['svm']['kernel_svm'],
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        n_estimators=10,  # Number of bagged models
        n_jobs=2,
        random_state=42
    )

    # Logistic Regression with increased regularization strength
    penalty = best_params['logreg'].get('penalty', 'l2')  # L1 or L2 penalty
    # Use 'saga' for L1, 'lbfgs' for L2
    solver = 'saga' if penalty == 'l1' else 'lbfgs'

    # Stronger regularization by reducing the C parameter (higher C = weaker regularization)
    meta_model = LogisticRegression(
        # Regularization strength (smaller C = stronger regularization)
        C=best_params['logreg']['C_logreg'],
        penalty=penalty,
        class_weight='balanced',
        random_state=42,
        solver=solver,
        max_iter=2000
    )

    # Stacking ensemble with Bagged SVM and XGBoost as base models
    stacking_model = StackingClassifier(
        estimators=[('bagged_svm', bagged_svm), ('xgb', xgb_model)],
        final_estimator=meta_model
    )

    # Train the ensemble model
    for _ in tqdm(range(1), desc="Training ensemble model"):
        stacking_model.fit(X_train, y_train)

    # Save the ensemble model
    joblib.dump(stacking_model, model_path)
    logging.info(f"Ensemble model trained and saved to {model_path}.\n")

    return stacking_model
