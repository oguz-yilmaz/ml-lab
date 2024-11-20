import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils import load_data, split_data


class LogisticRegression:
    """
    Lojistik Regresyon modeli sınıfı.

    Attributes:
        learning_rate (float): Öğrenme katsayısı
        num_iterations (int): Maksimum iterasyon sayısı
        weights (ndarray): Model ağırlıkları
        bias (float): Bias değeri
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
        Sigmoid aktivasyon fonksiyonu.

        Args:
            z (ndarray): logit fonksiyonu ciktisi

        Returns:
            skalar: Sigmoid fonksiyonu ciktisi
        """
        return np.round(1 / (1 + np.exp(-z)), 8)

    def cross_entropy_loss(self, y_true, y_pred_proba):
        """
        Cross entropy loss hesaplama.

        Args:
            y_true (skaler): Gerçek target degeri (0 veya 1)
            y_pred_proba (skaler): Tahmin edilen olasilik (0 ile 1 arasinda)

        Returns:
            float: Hesaplanan loss değeri
        """

        y_pred = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)  # log(0) olmaması için

        return -np.round(
            (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), 6
        )

    def stochastic_gradient_descent(self, X, y_true, y_predicted):
        """
        Stokastik gradyan algoritmasi. Agirliklari ve bias'i günceller.

        Args:
            X (ndarray): Ozellikler matrisi
            y_true (skalar): Etiket degeri
            y_predicted (skalar): Tahmin edilen olasilik degeri
        """

        # Gradient calculation
        dweight = (y_predicted - y_true) * X  # dL/dw = (y^ - y) * x
        dbias = y_predicted - y_true  # dL/db = y^ - y

        # Update weights and bias
        self.weights -= self.learning_rate * dweight  # w = w - eta * dL/dw
        self.bias -= self.learning_rate * dbias  # b = b - eta * dL/db

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Modeli egitme fonksiyonu.

        Args:
            X_train (ndarray): Egitim verileri
            y_train (ndarray): Egitim etiketleri
            X_val (ndarray): Dogrulama verileri
            y_val (ndarray): Dogrulama etiketleri
        """

        # (100, 2) -> 100 ornek, 2 ozellik (x1, x2)
        n_samples, n_features = X_train.shape

        # Veriyi standartlastir ki gradient descent iyi calissin
        # yoksa loss degerleri cok farkli cikiyor anlamsiz grafikler olusuyor
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

            # her bir ornegi al ve gradient descent ile agirliklari guncelle
            # asil egitim islemi burada yapiliyor
            for i in range(n_samples):
                current_x = standardized_X_train[i]
                current_y = y_train[i]

                z = np.dot(current_x, self.weights) + self.bias
                y_predicted = self.sigmoid(z)  # f(X, Q) = y^ = sigmoid(w*x + b)

                # Agirliklari ve bias'i guncelle
                self.stochastic_gradient_descent(current_x, current_y, y_predicted)

                epoch_training_loss += self.cross_entropy_loss(current_y, y_predicted)

            # Her epoch iterasyonu icin ortalama loss degerini kaydet
            self.training_loss.append(epoch_training_loss / n_samples)

            # Eger dogrulama verisi varsa, her epoch icin dogrulama loss degerini hesapla
            if X_val is not None and y_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    current_x = standardized_X_val[i]
                    current_y = y_val[i]

                    # agirliklari guncelleme, sadece sigmoid hesapla
                    z = np.dot(current_x, self.weights) + self.bias
                    y_predicted = self.sigmoid(z)  # f(X, Q) = y^ = sigmoid(w*x + b)

                    val_loss += self.cross_entropy_loss(current_y, y_predicted)

                # Her epoch icin dogrulama loss degerini kaydet
                self.validation_loss.append(val_loss / len(X_val))

    def predict_proba(self, X):
        """
        Olasilik tahminlerini hesapla.

        Args:
            X (ndarray): Ozellikler matrisi veya vektoru

        Returns:
            skaler: Tahmin edilen olasilik
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
        Sinif tahminlerini yap.
        Eger X, threshold'dan buyukse 1, degilse 0 dondur.

        Args:
            X (ndarray): Ozellikler matrisi veya vektoru
            threshold (float): Siniflandirma esik degeri

        Returns:
            ndarray: Tahmin edilen siniflar
        """

        probas = self.predict_proba(X)
        if isinstance(probas, (float, np.float64)):  # Single prediction
            return 1 if probas >= threshold else 0

        return (probas >= threshold).astype(int)  # Multiple predictions

    def standardize(self, X, fit=False):
        """
        Veriyi standartlastir.

        Args:
            X (ndarray): Ozellikler vektoru
            fit (bool): Eger True ise, ortalama ve standart sapma hesapla

        Returns:
            ndarray: Standartlastirilmis veri
        """

        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)

        return (X - self.mean) / self.std


def main():
    """
    Ana program fonksiyonu
    """

    # Veriyi yukle
    X, y = load_data("./dataset/hw1Data.txt")

    # Veriyi bol
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Modeli olustur ve egit
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train, X_val, y_val)

    # Save
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
