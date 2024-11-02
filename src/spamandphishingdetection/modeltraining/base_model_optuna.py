import os
import logging
import optuna
import joblib
import json
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'config.json')) as config_file:
    config = json.load(config_file)
    base_dir = config['base_dir']

# Define model_path and param_path
model_path = os.path.join(base_dir, 'additional_model_training', 'base_models_optuna')
param_path = os.path.join(base_dir, 'additional_model_training', 'base_models_optuna')


def conduct_optuna_study(X_train, y_train, model_name):
    """
    Conducts an Optuna study to find the best hyperparameters for the specified model.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data features.
    y_train : pandas.Series
        The training data labels.
    model_name : str
        The name of the model to optimize ('xgb', 'svm', 'logreg', 'rf', 'knn', 'lgbm').

    Returns
    -------
    dict
        The best hyperparameters for the specified model.
    """
    best_params = {}

    if model_name == 'XGBoost':
        # Optimize XGBoost parameters
        def xgb_objective(trial):
            n_estimators_xgb = trial.suggest_int('n_estimators_xgb', 50, 100)
            max_depth_xgb = trial.suggest_int('max_depth_xgb', 3, 10)
            learning_rate_xgb = trial.suggest_float(
                'learning_rate_xgb', 0.01, 0.3)
            reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
            reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

            model = XGBClassifier(
                n_estimators=n_estimators_xgb,
                max_depth=max_depth_xgb,
                learning_rate=learning_rate_xgb,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)

        logging.info("Starting Optuna study for XGBoost...")
        xgb_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        xgb_study.optimize(xgb_objective, n_trials=5)
        best_params = xgb_study.best_params

    elif model_name == 'SVM':
        # Optimize SVM parameters
        def svm_objective(trial):
            try:
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
                return 0

        logging.info("Starting Optuna study for SVM...")
        svm_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        svm_study.optimize(svm_objective, n_trials=5)
        best_params = svm_study.best_params
        # Rename the hyperparameter key to match SVC's expected parameter name
        best_params['C'] = best_params.pop('C_svm')
        best_params['kernel'] = best_params.pop('kernel_svm')

    elif model_name == 'Logistic Regression':
        # Optimize Logistic Regression parameters
        def logreg_objective(trial):
            try:
                C_logreg = trial.suggest_float('C_logreg', 0.0001, 1.0)
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

                model = LogisticRegression(
                    C=C_logreg,
                    penalty=penalty,
                    solver='saga' if penalty == 'l1' else 'lbfgs',
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000
                )
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                return accuracy_score(y_train, y_train_pred)
            except Exception as e:
                logging.error(
                    f"Error in Logistic Regression objective function: {e}")
                return 0

        logging.info("Starting Optuna study for Logistic Regression...")
        logreg_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        logreg_study.optimize(logreg_objective, n_trials=5)
        best_params = logreg_study.best_params

        # Rename the hyperparameter key to match LogisticRegression's expected parameter name
        best_params['C'] = best_params.pop('C_logreg')

    elif model_name == 'Random Forest':
        # Optimize Random Forest parameters
        def rf_objective(trial):
            n_estimators_rf = trial.suggest_int('n_estimators_rf', 50, 200)
            max_depth_rf = trial.suggest_int('max_depth_rf', 3, 20)
            min_samples_split_rf = trial.suggest_int(
                'min_samples_split_rf', 2, 10)
            min_samples_leaf_rf = trial.suggest_int(
                'min_samples_leaf_rf', 1, 10)

            model = RandomForestClassifier(
                n_estimators=n_estimators_rf,
                max_depth=max_depth_rf,
                min_samples_split=min_samples_split_rf,
                min_samples_leaf=min_samples_leaf_rf,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)

        logging.info("Starting Optuna study for Random Forest...")
        rf_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        rf_study.optimize(rf_objective, n_trials=5)
        best_params = rf_study.best_params
        # Rename the hyperparameter keys to match RandomForestClassifier's expected parameter names
        best_params['n_estimators'] = best_params.pop('n_estimators_rf')
        best_params['max_depth'] = best_params.pop('max_depth_rf')
        best_params['min_samples_split'] = best_params.pop(
            'min_samples_split_rf')
        best_params['min_samples_leaf'] = best_params.pop(
            'min_samples_leaf_rf')

    elif model_name == 'KNN':
        # Optimize KNN parameters
        def knn_objective(trial):
            n_neighbors_knn = trial.suggest_int('n_neighbors_knn', 1, 20)
            weights_knn = trial.suggest_categorical(
                'weights_knn', ['uniform', 'distance'])

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors_knn,
                weights=weights_knn
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)

        logging.info("Starting Optuna study for KNN...")
        knn_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        knn_study.optimize(knn_objective, n_trials=5)
        best_params = knn_study.best_params
        best_params['n_neighbors'] = best_params.pop('n_neighbors_knn')
        best_params['weights'] = best_params.pop('weights_knn')

    elif model_name == 'LightGBM':
        # Optimize LightGBM parameters
        def lgbm_objective(trial):
            num_leaves_lgbm = trial.suggest_int('num_leaves_lgbm', 20, 100)
            max_depth_lgbm = trial.suggest_int('max_depth_lgbm', 3, 20)
            learning_rate_lgbm = trial.suggest_float(
                'learning_rate_lgbm', 0.01, 0.3)
            n_estimators_lgbm = trial.suggest_int('n_estimators_lgbm', 50, 200)

            model = LGBMClassifier(
                num_leaves=num_leaves_lgbm,
                max_depth=max_depth_lgbm,
                learning_rate=learning_rate_lgbm,
                n_estimators=n_estimators_lgbm,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            return accuracy_score(y_train, y_train_pred)

        logging.info("Starting Optuna study for LightGBM...")
        lgbm_study = optuna.create_study(
            direction='maximize', sampler=TPESampler())
        lgbm_study.optimize(lgbm_objective, n_trials=5)
        best_params = lgbm_study.best_params
        best_params['num_leaves'] = best_params.pop('num_leaves_lgbm')
        best_params['max_depth'] = best_params.pop('max_depth_lgbm')
        best_params['learning_rate'] = best_params.pop('learning_rate_lgbm')
        best_params['n_estimators'] = best_params.pop('n_estimators_lgbm')

    # Save the best parameters to a file
    param_file_path = os.path.join(
        param_path, f"{model_name}_best_params.json")
    with open(param_file_path, 'w') as param_file:
        json.dump(best_params, param_file)

    logging.info(f"Best parameters for {model_name}: {best_params}")
    return best_params


def model_training(X_train, y_train, X_test, y_test, model, model_name):
    # Conduct Optuna study based on the model name
    logging.info(f"Conducting Optuna study for {model_name}...")
    best_params = conduct_optuna_study(X_train, y_train, model_name)

    # Initialize the model with the best hyperparameters
    if model_name == 'XGBoost':
        model = XGBClassifier(
            **best_params, eval_metric='mlogloss', random_state=42)
    elif model_name == 'SVM':
        model = SVC(**best_params, probability=True,
                    class_weight='balanced', random_state=42)
    elif model_name == 'Logistic Regression':
        solver = 'saga' if best_params['penalty'] == 'l1' else 'lbfgs'
        model = LogisticRegression(
            **best_params, solver=solver, class_weight='balanced', random_state=42, max_iter=2000)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(**best_params, random_state=42)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(**best_params)
    elif model_name == 'LightGBM':
        model = LGBMClassifier(**best_params, random_state=42)

    # Train the model with the training data
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

    joblib.dump(model, os.path.join(model_path, f"{model_name}.pkl"))
    return model, test_accuracy
