import pickle

import matplotlib.pyplot as plt

from train import LogisticRegression
from utils import load_data


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
    plt.show()


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
    plt.show()


def main():
    model = None

    X, y = load_data("./dataset/hw1Data.txt")

    # Modeli yükle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    plot_data(X, y)
    plot_loss_curves(model.training_loss, model.validation_loss)


if __name__ == "__main__":
    main()
