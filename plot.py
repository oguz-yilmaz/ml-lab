import pickle

import matplotlib.pyplot as plt
import numpy as np

from train import LogisticRegression
from utils import calculate_metrics, load_data, split_data


def plot_data(X, y):
    """
    Veri setini görselleştir.

    Args:
        X (ndarray): Özellikler (2D)
        y (ndarray): Etiketler (0 veya 1D)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Ret")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Kabul")
    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.title("İş Başvurusu Sonuçları")
    plt.legend()

    plt.savefig("./plots/data.png")
    plt.close()


def plot_loss_curves(training_loss, validation_loss):
    """
    Eğitim ve doğrulama loss değerlerini görselleştir.

    Args:
        training_loss (list): Eğitim loss değerleri
        validation_loss (list): Doğrulama loss değerleri
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label="Eğitim Loss")
    plt.plot(validation_loss, label="Doğrulama Loss")
    plt.xlabel("İterasyon")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Eğitim ve Doğrulama Loss Değerleri")
    plt.legend()

    plt.savefig("./plots/loss_curves.png")
    plt.close()


def plot_predictions(X, y, y_pred, title):
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

    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.title(title)
    plt.legend()

    plt.savefig(f"./plots/{title}.png")
    plt.close()


def main():
    model = None

    X, y = load_data("./dataset/hw1Data.txt")

    # Modeli yükle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Veriyi bol
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    plot_data(X, y)
    plot_loss_curves(model.training_loss, model.validation_loss)

    sets = [
        ("Eğitim", X_train, y_train),
        ("Doğrulama", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    for name, X_set, y_set in sets:
        y_pred = model.predict(X_set)
        plot_predictions(X_set, y_set, y_pred, f"{name} Seti Tahminleri")
        accuracy, precision, recall, f1 = calculate_metrics(y_set, y_pred)

        print(f"\n{name} Seti Metrikleri:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    print("\nGörselleştirmeler kaydedildi. Plots klasörüne bakınız.")


if __name__ == "__main__":
    main()
