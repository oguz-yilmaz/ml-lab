import pickle

import numpy as np

from train import LogisticRegression
from utils import calculate_metrics, load_data, split_data


def main():
    model = None
    X, y = load_data("./dataset/hw1Data.txt")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    sets = [
        ("Eğitim", X_train, y_train),
        ("Doğrulama", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    for name, X_set, y_set in sets:
        y_pred = model.predict(X_set)
        accuracy, precision, recall, f1 = calculate_metrics(y_set, y_pred)

        print(f"\n Metrics for {name} Set")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
