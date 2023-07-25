import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def sigmoid(z):
    """
    This function acts as the sigmoid function
    :param z: Input
    :return: Value of the function
    """
    return np.exp(-np.logaddexp(0, -z))


def sigmoid_der(z):
    """
    This function acts as the derivative of the sigmoid function
    :param z: Input
    :return: Value of the derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


def dimensions(X, y):
    """
    This function gives the minimum number of neurons needed in each layer
    :param X: Input matrix
    :param y: Label matrix
    :return: Number of neurons in input, hidden and output layers
    """
    dim_input = X.shape[0]
    dim_hidden = 4
    dim_output = y.shape[0]

    return dim_input, dim_hidden, dim_output


def init_parameters(input_dim, hidden_layer_dim, output_dim):
    """
    This function initialises the weights and biases
    :param input_dim: Number of neurons in input layer
    :param hidden_layer_dim: Number of neurons in hidden layer
    :param output_dim: Number of neurons in output layer
    :return: Initialised weights and biases
    """
    np.random.seed(42)  # Setting seed to get consistent results
    W1 = np.random.uniform(-0.5, 0.5, (hidden_layer_dim, input_dim))
    b1 = np.zeros((hidden_layer_dim, 1))
    W2 = np.random.uniform(-0.5, 0.5, (output_dim, hidden_layer_dim))
    b2 = np.zeros((output_dim, 1))

    return W1, b1, W2, b2


def forward_prop(X, W1, b1, W2, b2):
    """
    This function performs the forward propagation calculations
    :param X: Input matrix
    :param W1: Initialised weight matrix of the first layer
    :param b1: Initialised bias matrix of the first layer
    :param W2: Initialised weight matrix of the second layer
    :param b2: Initialised bias matrix of the second layer
    :return: Activated values of each layer
    """
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return A1, A2


def back_prop(X, y, W2, A1, A2):
    """
    This function performs the backpropagation calculations
    :param X: Input matrix
    :param y: Label matrix
    :param W2: Initialised weight matrix of the second layer
    :param A1: Activated value matrix of the first layer
    :param A2: Activated value matrix of the second layer
    :return: Gradients of weights and biases
    """
    m = X.shape[1]

    # Calculating gradients
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    This function updates weights and biases from gradients
    :param W1: Initialised weight matrix of the first layer
    :param b1: Initialised bias matrix of the first layer
    :param W2: Initialised weight matrix of the second layer
    :param b2: Initialised bias matrix of the second layer
    :param dW1: Gradient of the weight matrix of the first layer
    :param db1: Gradient of the bias matrix of the first layer
    :param dW2: Gradient of the weight matrix of the second layer
    :param db2: Gradient of the bias matrix of the second layer
    :param learning_rate: Learning rate
    :return: Updated weights and biases
    """
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


def predict(X, W1_u, b1_u, W2_u, b2_u):
    """
    This function predicts the labels
    :param X: Input matrix
    :param W1_u: Updated weight matrix of the first layer
    :param b1_u: Updated bias matrix of the first layer
    :param W2_u: Updated weight matrix of the second layer
    :param b2_u: Updated bias matrix of the second layer
    :return:
    """
    A1, A2 = forward_prop(X, W1_u, b1_u, W2_u, b2_u)
    return (A2 >= 0.5).astype(int)  # Makes all the values greater than or equal to 0.5 as 1 and the rest as 0


def train_neural_network(X, y, dim_hidden=4, num_epochs=1000, learning_rate=0.003):
    """
    This function trains the Neural Network on the given dataset
    :param X: Input matrix
    :param y: Label matrix
    :param dim_hidden: Number of neurons in the hidden layer
    :param num_epochs: Number of iterations
    :param learning_rate: Learning rate
    :return: Updated weights and biases
    """
    np.random.seed(42)  # setting seed to get consistent results
    dim_input, _, dim_output = dimensions(X, y)
    W1, b1, W2, b2 = init_parameters(dim_input, dim_hidden, dim_output)

    for _ in range(num_epochs):
        A1, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_prop(X, y, W2, A1, A2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    return W1, b1, W2, b2


def accuracy(y_true, y_pred):
    """
    This function gives the accuracy of the predicted outcomes
    :param y_true: Label matrix
    :param y_pred: Predicted outcomes matrix
    :return: Accuracy
    """
    return np.mean(y_true == y_pred)  # Computes mean of the number of equal/correct outcomes to total outcomes =>
    # accuracy


def tune_hyperparameter(X_train, y_train, X_test, y_test, learning_rates_list, hidden_sizes_list):
    """
    This function tunes the hyperparameters 'learning rate' and 'number of neurons in hidden layer'
    :param X_train: Input matrix from training dataset
    :param y_train: Label matrix from training dataset
    :param X_test: Input matrix from testing dataset
    :param y_test: Label matrix from testing dataset
    :param learning_rates_list: List of various learning rates
    :param hidden_sizes_list: List of various number of neurons in hidden layer
    :return: Learning rate and the number of neurons in hidden layer with best achieved accuracy
    """
    best_accuracy = 0
    best_lr = None
    best_hidden_size = None

    # Compute accuracy for each number of neurons in the list for every learning rate in the list
    for lr in learning_rates_list:
        for hidden_size in hidden_sizes_list:
            W1, b1, W2, b2 = train_neural_network(X_train, y_train, dim_hidden=hidden_size, learning_rate=lr)
            y_pred = predict(X_test, W1, b1, W2, b2)
            acc = accuracy(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_lr = lr
                best_hidden_size = hidden_size

    return best_lr, best_hidden_size, best_accuracy


# Loading training datasets
train_2 = pd.read_csv('data/ds2_train.csv')
X_train_2, y_train_2 = train_2[['x_1', 'x_2']], train_2['y']

# Loading testing datasets
test_2 = pd.read_csv('data/ds2_test.csv')
X_test_2, y_test_2 = test_2[['x_1', 'x_2']], test_2['y']

# Changing shapes of the datas to fit the learning algorithms
# Instead of passing each sample of size (2, 1) one by one, we can pass the whole data at once
X_train_2_scratch = X_train_2.T  # Change (800,2) to (2, 800)
X_test_2_scratch = X_test_2.T  # Change (100, 2) to (2, 100)
y_train_2_scratch = np.array(y_train_2).reshape(-1, 1).T  # Change (800,) to (1, 800)
y_test_2_scratch = np.array(y_test_2).reshape(-1, 1).T  # Change (100,) to (1, 100)

# Subtask 2: Training and Prediction - First Training Set
# Training the Neural Network - Training set
W_train_1, b_train_1, W_train_2, b_train_2 = train_neural_network(X_train_2_scratch, y_train_2_scratch)

# Predicting outcomes - Training set and Testing set
y_pred_train_scratch_2 = predict(X_train_2_scratch, W_train_1, b_train_1, W_train_2, b_train_2)
y_pred_test_scratch_2 = predict(X_test_2_scratch, W_train_1, b_train_1, W_train_2, b_train_2)

# Getting accuracies - Training set and Testing set
accuracy_train_scratch_2 = accuracy(y_train_2_scratch, y_pred_train_scratch_2)
accuracy_test_scratch_2 = accuracy(y_test_2_scratch, y_pred_test_scratch_2)

# Printing accuracies - Training set and Testing set
print("Accuracy on Training Set 2 (Scratch):", accuracy_train_scratch_2)
print("Accuracy on Test Set 2 (Scratch):", accuracy_test_scratch_2, "\n")

# Subtask 3: Hyperparameter Tuning
learning_rates = [0.004, 0.003, 0.002, 0.001]
hidden_sizes = [2, 4, 8]

# Printing the best learning rate, best number of neurons in hidden layer and its accuracy
best_lr_2, best_hidden_size_2, best_accuracy_2 = tune_hyperparameter(X_train_2_scratch, y_train_2_scratch,
                                                                     X_test_2_scratch, y_test_2_scratch,
                                                                     learning_rates, hidden_sizes)
print("Best learning rate for Task 2:", best_lr_2)
print("Best hidden size for Task 2:", best_hidden_size_2)
print("Best accuracy with tuned hyperparameter for Task 2:", best_accuracy_2, "\n")

# Subtask 4: Comparison with Scikit-Learn
# Initialising and training MLPClassifier - Training set
mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42, learning_rate_init=0.003)
mlp.fit(X_train_2, y_train_2)

# Predicting outcomes - Training set and Testing set
y_pred_train_mlp_2 = mlp.predict(X_train_2)
y_pred_test_mlp_2 = mlp.predict(X_test_2)

# Getting accuracies - Training set and Testing set
accuracy_train_mlp_2 = accuracy_score(y_train_2, y_pred_train_mlp_2)
accuracy_test_mlp_2 = accuracy_score(y_test_2, y_pred_test_mlp_2)

# Printing accuracies - Training set and Testing set
print("Accuracy on Training Set 2 (Scikit-Learn):", accuracy_train_mlp_2)
print("Accuracy on Test Set 2 (Scikit-Learn):", accuracy_test_mlp_2)
