import pickle

import numpy as np

from train import LogisticRegression
from utils import load_data, split_data


def calculate_metrics(y_true, y_pred):
    """
    Performans metriklerini hesapla.

    Args:
        y_true (ndarray): Gerçek etiketler
        y_pred (ndarray): Tahmin edilen etiketler

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


def main():
    model = None
    X, y = load_data("./dataset/hw1Data.txt")

    # Veriyi bol
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Modeli yükle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Model:", model)

    # Tahminler yap ve metrikleri hesapla
    sets = [
        ("Eğitim", X_train, y_train),
        ("Doğrulama", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    for name, X_set, y_set in sets:
        y_pred = model.predict(X_set)
        accuracy, precision, recall, f1 = calculate_metrics(y_set, y_pred)

        print(f"\n{name} Seti Metrikleri:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")


if __name__ == "__main__":
    main()
