import numpy as np


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.

    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels

    Returns:
        tuple: accuracy, precision, recall, f1_score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return accuracy, precision, recall, f1_score


def load_data(filename):
    """
    Load the data file and separate features and labels.

    Args:
        filename (str): Path to the data file

    Returns:
        tuple: Features (X) and labels (y)
    """

    # data = array([[34.62365962, 78.02469282,  0.        ],
    #               [30.28671077, 43.89499752,  0.        ],
    #               [35.84740877, 72.90219803,  0.        ]
    #               ...])
    data = np.loadtxt(filename, delimiter=",")
    # X = array([[34.62365962, 78.02469282],
    #            [30.28671077, 43.89499752],
    #            [35.84740877, 72.90219803],
    #            ...])
    X = data[:, :2]
    # y = array([0., 0., 0., 1., 1., 0., 1., 1., 1., 1., ...])
    y = data[:, 2]

    return X, y


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Split the data into training, validation, and test sets.

    Args:
        X (ndarray): Features
        y (ndarray): Labels
        train_ratio (float): Proportion of training set
        val_ratio (float): Proportion of validation set

    Returns:
        tuple: Split data sets
    """
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return X_train, y_train, X_val, y_val, X_test, y_test
