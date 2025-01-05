import pickle

import matplotlib.pyplot as plt
import numpy as np

from train import LogisticRegression
from utils import calculate_metrics, load_data, split_data


def plot_data(X, y):
    """
    Visualize the dataset.

    Args:
        X (ndarray): Features (2D)
        y (ndarray): Labels
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Rejected")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Accepted")
    plt.xlabel("1st Exam Score")
    plt.ylabel("2nd Exam Score")
    plt.title("Job Application Results")
    plt.legend()

    plt.savefig("./plots/data.png")
    plt.close()


def plot_loss_curves(training_loss, validation_loss):
    """
    Visualize the training and validation loss values.

    Args:
        training_loss (list): Training loss values
        validation_loss (list): Validation loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Loss Values")
    plt.legend()

    plt.savefig("./plots/loss_curves.png")
    plt.close()


def plot_predictions(X, y, y_pred, title, save_name):
    plt.figure(figsize=(10, 6))

    # Plot correct predictions
    correct_0 = (y == 0) & (y_pred == 0)
    correct_1 = (y == 1) & (y_pred == 1)
    plt.scatter(
        X[correct_0][:, 0],
        X[correct_0][:, 1],
        color="green",
        marker="o",
        label="Correct Negative",
    )
    plt.scatter(
        X[correct_1][:, 0],
        X[correct_1][:, 1],
        color="blue",
        marker="o",
        label="Correct Positive",
    )

    # Plot incorrect predictions
    wrong_0 = (y == 0) & (y_pred == 1)  # False Positive
    wrong_1 = (y == 1) & (y_pred == 0)  # False Negative
    plt.scatter(
        X[wrong_0][:, 0],
        X[wrong_0][:, 1],
        color="red",
        marker="x",
        label="False Positive",
    )
    plt.scatter(
        X[wrong_1][:, 0],
        X[wrong_1][:, 1],
        color="orange",
        marker="x",
        label="False Negative",
    )

    plt.xlabel("1st Exam Score")
    plt.ylabel("2nd Exam Score")
    plt.title(title)
    plt.legend()

    plt.savefig(f"./plots/{save_name}.png")
    plt.close()


def main():
    model = None

    X, y = load_data("./dataset/hw1Data.txt")

    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    plot_data(X, y)
    plot_loss_curves(model.training_loss, model.validation_loss)

    sets = [
        ("Training", X_train, y_train, "training"),
        ("Validation", X_val, y_val, "validation"),
        ("Test", X_test, y_test, "test"),
    ]

    for name, X_set, y_set, save_name in sets:
        y_pred = model.predict(X_set)
        plot_predictions(
            X_set, y_set, y_pred, f"{name} Set Predictions", f"{save_name}_predictions"
        )
        accuracy, precision, recall, f1 = calculate_metrics(y_set, y_pred)

        print(f"\n{name} Set Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    print("\nVisualizations saved. Please check the plots folder.")


if __name__ == "__main__":
    main()
