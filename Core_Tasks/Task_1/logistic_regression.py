import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def sigmoid(z):
    """
    This function acts as the sigmoid function
    :param z: Input
    :return: Value of the function
    """
    return np.exp(-np.logaddexp(0, -z))


def fit_logistic_regression(X, y, learning_rate=0.75, num_epochs=1000):
    """
    This function fits Logistic Regression to a given dataset
    :param X: Input dataset
    :param y: Labels dataset
    :param learning_rate: Learning rate to control gradient descent
    :param num_epochs: Number of iterations
    :return: Fitted weights and bias
    """
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)  # initialises a weight matrix with zeros
    bias = 0

    for _ in range(num_epochs):
        # Producing predicted outcomes
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Calculating gradients
        dW = (1 / num_samples) * np.dot(X.T, (y_pred - y))
        dB = (1 / num_samples) * np.sum(y_pred - y)

        # Updating parameters
        weights -= learning_rate * dW
        bias -= learning_rate * dB

    return weights, bias


def predict(X, weights, bias):
    """
    This function predicts the labels
    :param X: Input dataset
    :param weights: Fitted weights matrix
    :param bias: Fitted weights matrix
    :return: Predicted outcomes matrix (one-hot encoded)
    """
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)  # Makes all the values greater than or equal to 0.5 as 1 and the rest as 0


def accuracy(y_true, y_pred):
    """
    This function gives the accuracy of the predicted outcomes
    :param y_true: Label matrix
    :param y_pred: Predicted outcomes matrix (one-hot encoded)
    :return: Accuracy
    """
    return np.mean(y_true == y_pred)  # Computes mean of the number of equal/correct outcomes to total outcomes =>
    # accuracy


def tune_hyperparameter(X_train, y_train, X_test, y_test, learning_rates_list):
    """
    This function tunes the hyperparameter 'Learning rate'
    :param X_train: Input from training dataset
    :param y_train: Labels from training dataset
    :param X_test: Input from testing dataset
    :param y_test: Labels from testing dataset
    :param learning_rates_list: List of different learning rates
    :return: Learning rate with best achieved accuracy
    """
    best_accuracy = 0
    best_lr = None

    for lr in learning_rates_list:
        weights, bias = fit_logistic_regression(X_train, y_train, learning_rate=lr)
        y_pred = predict(X_test, weights, bias)
        acc = accuracy(y_test, y_pred)

        if acc > best_accuracy:
            best_accuracy = acc
            best_lr = lr

    return best_lr, best_accuracy


# Loading training datasets
train_1 = pd.read_csv('data/ds1_train.csv')
X_train_1, y_train_1 = train_1[['x_1', 'x_2']], train_1['y']


# Loading testing datasets
test_1 = pd.read_csv('data/ds1_test.csv')
X_test_1, y_test_1 = test_1[['x_1', 'x_2']], test_1['y']


# Subtask 2: Training and Prediction - First Training Set
# Fitting logistic function - Training set
weights_1, bias_1 = fit_logistic_regression(X_train_1, y_train_1, learning_rate=0.75)

# Predicting outcomes - Training set and Testing set
y_pred_train_scratch_1 = predict(X_train_1, weights_1, bias_1)
y_pred_test_scratch_1 = predict(X_test_1, weights_1, bias_1)

# Getting accuracies - Training set and Testing set
accuracy_train_scratch_1 = accuracy(y_train_1, y_pred_train_scratch_1)
accuracy_test_scratch_1 = accuracy(y_test_1, y_pred_test_scratch_1)

# Printing accuracies - Training set and Testing set
print("Accuracy on Training Set 1 (Scratch):", accuracy_train_scratch_1)
print("Accuracy on Test Set 1 (Scratch):", accuracy_test_scratch_1, "\n")


# Subtask 3: Hyperparameter Tuning
learning_rates = [1.0, 0.95, 0.9, 0.87, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.39, 0.35, 0.3, 0.25,
                  0.2, 0.15, 0.1, 0.05, 0.01]

# Printing best learning rate and its accuracy
best_lr_1, best_accuracy_1 = tune_hyperparameter(X_train_1, y_train_1, X_test_1, y_test_1, learning_rates)
print("Best learning rate for Task 1:", best_lr_1)
print("Best accuracy with tuned hyperparameter for Task 1:", best_accuracy_1, "\n")

# Subtask 4: Comparison with Scikit-Learn
# Fitting logistic function - Training set
model_sklearn = LogisticRegression()
model_sklearn.fit(X_train_1, y_train_1)

# Predicting outcomes - Training set and Testing set
y_pred_train_sklearn_1 = model_sklearn.predict(X_train_1)
y_pred_test_sklearn_1 = model_sklearn.predict(X_test_1)

# Get accuracies - Training set and Testing set
accuracy_train_sklearn_1 = accuracy_score(y_train_1, y_pred_train_sklearn_1)
accuracy_test_sklearn_1 = accuracy_score(y_test_1, y_pred_test_sklearn_1)

# Printing Accuracies
print("Accuracy on Training Set 1 (Scikit-Learn):", accuracy_train_sklearn_1)
print("Accuracy on Test Set 1 (Scikit-Learn):", accuracy_test_sklearn_1)
