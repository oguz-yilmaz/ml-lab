import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    """
    Lojistik Regresyon sınıflandırıcı sınıfı.

    Attributes:
        learning_rate (float): Öğrenme katsayısı
        num_iterations (int): Maksimum iterasyon sayısı
        weights (ndarray): Model ağırlıkları
        bias (float): Bias değeri
    """

    def __init__(self, learning_rate=0.01, num_iterations=10):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.training_loss = []
        self.validation_loss = []
        self.mean = None
        self.std = None
        self.standardize_inputs = False

    # ✅ Ok
    def sigmoid(self, z):
        """
        Sigmoid aktivasyon fonksiyonu.

        Args:
            z (ndarray): Girdi değeri

        Returns:
            skalar: Sigmoid fonksiyonu çıktısı
        """
        return np.round(1 / (1 + np.exp(-z)), 8)

    # ✅ Ok
    # infinite when: y_pred_proba = 0 and y_true = 1
    # infinite when: y_pred_proba = 1 and y_true = 0
    def cross_entropy_loss(self, y_true, y_pred_proba):
        """
        Cross entropy loss hesaplama.

        Args:
            y_true (scalar): Gerçek etiket (0 veya 1)
            y_pred (scalar): Tahmin edilen olasılık (0 ile 1 arasında)

        Returns:
            float: Hesaplanan loss değeri
        """
        y_pred = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)  # log(0) olmaması için
        print(f"y_true: {y_true}, y_pred_proba: {y_pred}, y_pred: {y_pred}")
        return -np.round(
            (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), 6
        )

    # ✅ Ok
    def stochastic_gradient_descent(self, X, y_true, y_predicted):
        """
        Stokastik gradyan algoritması. Ağırlıkları ve bias'ı günceller.

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
        Modeli eğitme fonksiyonu.

        Args:
            X_train (ndarray): Eğitim verileri
            y_train (ndarray): Eğitim etiketleri
            X_val (ndarray): Doğrulama verileri
            y_val (ndarray): Doğrulama etiketleri
        """

        print(f"X_val: {X_val}, y_val: {y_val}")
        print("---------------------------------")
        print(f"X_train: {X_train}, y_train: {y_train}")

        # n_samples: number of training examples (number of rows in X_train)
        # n_features: number of input features/variables (number of columns in X_train)
        n_samples, n_features = X_train.shape

        # Standardize the data so that the gradient descent works well
        standardized_X_train = (
            self.standardize(X_train, fit=True) if self.standardize_inputs else X_train
        )
        standardized_X_val = (
            self.standardize(X_val) if self.standardize_inputs else X_val
        )

        # Initialize weights and bias here
        self.weights = np.zeros(n_features)
        self.bias = 0

        train_loss = 0

        # epoch
        for iteration in range(self.num_iterations):
            epoch_training_loss = 0

            # Process each training example
            for i in range(n_samples):
                current_x = standardized_X_train[i]  # Get current row (x1, x2)
                current_y = y_train[i]  # Get current label

                z = np.dot(current_x, self.weights) + self.bias
                y_predicted = self.sigmoid(z)  # f(X, Q) = y^ = sigmoid(w*x + b)

                # print(
                #     f"Current x: {current_x}, Current y: {current_y}, z: {z}, y_predicted: {y_predicted}"
                # )

                # Forward pass and update weights
                self.stochastic_gradient_descent(current_x, current_y, y_predicted)

                # Calculate and store training loss
                train_loss_current = self.cross_entropy_loss(current_y, y_predicted)
                epoch_training_loss += train_loss_current
                # print(f"Current train loss_calculated: {train_loss_current}")
                # print(
                #     f"Current y_true: {current_y}, Current y_predicted: {y_predicted}"
                # )
            self.training_loss.append(epoch_training_loss / n_samples)

            # Validation loop (if validation data provided)
            if X_val is not None and y_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    current_x = standardized_X_val[i]
                    current_y = y_val[i]

                    # Only forward pass for validation (no weight updates)
                    z = np.dot(current_x, self.weights) + self.bias
                    y_predicted = self.sigmoid(z)  # f(X, Q) = y^ = sigmoid(w*x + b)

                    # Calculate validation loss
                    val_loss += self.cross_entropy_loss(current_y, y_predicted)
                    # Store average validation loss
                self.validation_loss.append(val_loss / len(X_val))

    def predict_proba(self, x):
        """
        Olasılık tahminlerini hesapla.

        Args:
            X (ndarray): Ozellikler vektörü

        Returns:
            skalar: Tahmin edilen olasılık
        """
        linear_model = np.dot(x, self.weights) + self.bias
        return self.sigmoid(linear_model)

    # ✅ Ok
    def predict(self, x, threshold=0.5):
        """
        Sınıf tahminlerini yap.
        If X > threshold, return 1, else return 0.

        Args:
            X (ndarray): Girdi verileri
            threshold (float): Sınıflandırma eşik değeri

        Returns:
            ndarray: Tahmin edilen sınıflar
        """

        proba = self.predict_proba(x)  # Scalar value
        return 1 if proba >= threshold else 0

    def standardize(self, X, fit=False):
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)

        return (X - self.mean) / self.std


def load_data(filename):
    """
    Veri dosyasını yükle ve özellikleri ile etiketleri ayır.

    Args:
        filename (str): Veri dosyasının yolu

    Returns:
        tuple: Özellikler (X) ve etiketler (y)
    """

    # data = array([[34.62365962, 78.02469282,  0.        ],
    #               [30.28671077, 43.89499752,  0.        ],
    #               [35.84740877, 72.90219803,  0.        ]
    #               ...])
    data = np.loadtxt(filename, delimiter=",")
    # X = array([[34.62365962, 78.02469282],
    #           [30.28671077, 43.89499752],
    #           [35.84740877, 72.90219803],
    #           ...])
    X = data[:, :2]
    # y = array([0., 0., 0., 1., 1., 0., 1., 1., 1., 1., ...])
    y = data[:, 2]

    return X, y


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Veriyi eğitim, doğrulama ve test setlerine böl.

    Args:
        X (ndarray): Özellikler
        y (ndarray): Etiketler
        train_ratio (float): Eğitim seti oranı
        val_ratio (float): Doğrulama seti oranı

    Returns:
        tuple: Bölünmüş veri setleri
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
    """
    Ana program fonksiyonu
    """

    enable_plot = True

    # Veriyi yükle
    X, y = load_data("./dataset/hw1Data.txt")
    print(X.shape, y.shape)

    # Veriyi böl
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Veriyi görselleştir
    if enable_plot:
        plot_data(X, y)

    # Modeli oluştur ve eğit
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train, X_val, y_val)

    print(f"loss training length: {len(model.training_loss)}")
    print("--------------------------------------")
    # print(f"loss training: {model.training_loss}")

    print(f"Loss training: {model.training_loss}")
    print(f"Loss validation: {model.validation_loss}")

    # Loss grafiklerini çiz
    if enable_plot:
        plot_loss_curves(model.training_loss, model.validation_loss)

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
