import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils import load_data, split_data


class LogisticRegression:
    """
    Logistic Regression model class.

    Attributes:
        learning_rate (float): Learning rate
        num_iterations (int): Maximum number of iterations
        weights (ndarray): Model weights
        bias (float): Bias value
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.training_loss = []
        self.validation_loss = []
        self.mean = None
        self.std = None
        self.standardize_inputs = True

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Args:
            z (ndarray): Output of the logit function

        Returns:
            scalar: Output of the sigmoid function
        """
        return np.round(1 / (1 + np.exp(-z)), 8)

    def cross_entropy_loss(self, y_true, y_pred_proba):
        """
        Calculate cross entropy loss.

        Args:
            y_true (scalar): True target value (0 or 1)
            y_pred_proba (scalar): Predicted probability (between 0 and 1)

        Returns:
            float: Calculated loss value
        """

        y_pred = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)  # to avoid log(0)

        return -np.round(
            (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), 6
        )

    def stochastic_gradient_descent(self, X, y_true, y_predicted):
        """
        Stochastic gradient algorithm. Updates weights and bias.

        Args:
            X (ndarray): Feature matrix
            y_true (scalar): Label value
            y_predicted (scalar): Predicted probability
        """

        # Gradient calculation
        dweight = (y_predicted - y_true) * X  # dL/dw = (y_hat - y) * x
        dbias = y_predicted - y_true  # dL/db = y_hat - y

        # Update weights and bias
        self.weights -= self.learning_rate * dweight  # w = w - eta * dL/dw
        self.bias -= self.learning_rate * dbias  # b = b - eta * dL/db

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Model training function.

        Args:
            X_train (ndarray): Training data
            y_train (ndarray): Training labels
            X_val (ndarray): Validation data
            y_val (ndarray): Validation labels
        """

        # (100, 2) -> 100 samples, 2 features (x1, x2)
        n_samples, n_features = X_train.shape

        # Standardize the data so that gradient descent works better.
        # Otherwise, the loss values may vary widely and produce meaningless plots.
        standardized_X_train = (
            self.standardize(X_train, fit=True) if self.standardize_inputs else X_train
        )
        standardized_X_val = (
            self.standardize(X_val) if self.standardize_inputs else X_val
        )

        self.weights = np.zeros(n_features)
        self.bias = 0

        # epoch
        for iteration in range(self.num_iterations):
            epoch_training_loss = 0

            # Take each sample and update weights using gradient descent
            # The actual training process happens here
            for i in range(n_samples):
                current_x = standardized_X_train[i]
                current_y = y_train[i]

                z = np.dot(current_x, self.weights) + self.bias
                y_predicted = self.sigmoid(z)  # f(X, Q) = y_hat = sigmoid(w*x + b)

                # Update weights and bias
                self.stochastic_gradient_descent(current_x, current_y, y_predicted)

                epoch_training_loss += self.cross_entropy_loss(current_y, y_predicted)

            # For each epoch iteration, record the average loss value
            self.training_loss.append(epoch_training_loss / n_samples)

            # If validation data is available, compute validation loss for each epoch
            if X_val is not None and y_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    current_x = standardized_X_val[i]
                    current_y = y_val[i]

                    # Do not update weights, just compute sigmoid
                    z = np.dot(current_x, self.weights) + self.bias
                    y_predicted = self.sigmoid(z)  # f(X, Q) = y_hat = sigmoid(w*x + b)

                    val_loss += self.cross_entropy_loss(current_y, y_predicted)

                # Record the validation loss value for each epoch
                self.validation_loss.append(val_loss / len(X_val))

    def predict_proba(self, X):
        """
        Calculate probability predictions.

        Args:
            X (ndarray): Feature matrix or vector

        Returns:
            scalar: Predicted probability
        """

        if self.standardize_inputs:
            if X.ndim == 1:
                X = (X - self.mean) / self.std
            else:
                X = np.array([(x - self.mean) / self.std for x in X])

        if X.ndim == 1:
            linear_model = np.dot(X, self.weights) + self.bias
            return self.sigmoid(linear_model)

        return np.array([self.sigmoid(np.dot(x, self.weights) + self.bias) for x in X])

    def predict(self, X, threshold=0.5):
        """
        Make class predictions.
        If X >= threshold, return 1; otherwise, 0.

        Args:
            X (ndarray): Feature matrix or vector
            threshold (float): Classification threshold

        Returns:
            ndarray: Predicted classes
        """

        probas = self.predict_proba(X)
        if isinstance(probas, (float, np.float64)):  # Single prediction
            return 1 if probas >= threshold else 0

        return (probas >= threshold).astype(int)  # Multiple predictions

    def standardize(self, X, fit=False):
        """
        Standardize the data.

        Args:
            X (ndarray): Feature vector
            fit (bool): If True, compute mean and standard deviation

        Returns:
            ndarray: Standardized data
        """

        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)

        return (X - self.mean) / self.std


def main():
    """
    Main program function
    """

    # Load data
    X, y = load_data("./dataset/hw1Data.txt")

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Create and train the model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train, X_val, y_val)

    # Save
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
