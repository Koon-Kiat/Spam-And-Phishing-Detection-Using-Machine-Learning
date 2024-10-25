import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=6, n_jobs=3, train_sizes=np.linspace(0.1, 1.0, 6)):
    """
    Plots the learning curve for the provided estimator.

    Args:
        estimator (object): The estimator to plot the learning curve for.
        X (pd.DataFrame): The training data features.
        y (pd.Series): The training data labels.
        title (str): The title of the plot.
        ylim (tuple): The y-axis limits for the plot.
        cv (int): The number of cross-validation folds.
        n_jobs (int): The number of jobs to run in parallel.
        train_sizes (array): The sizes of the training sets.

    Returns:
        None
    """
    logging.info("Starting the plot_learning_curve function.")

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    try:
        logging.info("Calling sklearn's learning_curve function.")
        train_sizes, train_scores, valid_scores = learning_curve(
            estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)
        logging.info("learning_curve function executed successfully.")

        # Plot the learning curves
        plt.plot(train_sizes, train_scores.mean(axis=1),
                 'o-', color='r', label='Training score')
        plt.plot(train_sizes, valid_scores.mean(axis=1),
                 'o-', color='g', label='Validation score')
        plt.legend(loc='best')
        plt.grid()
        logging.info("Plotting the learning curve.")
        plt.show()
        logging.info("Learning curve plot displayed successfully.\n")
    except Exception as e:
        logging.error(
            f"An error occurred while plotting the learning curve: {e}")
