import numpy as np
import matplotlib.pyplot as plt

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

x_train = x_train.reshape(60000, 784).T / 255.0
x_test = x_test.reshape(10000, 784).T / 255.0
y_train = y_train.reshape(1, 60000)
y_test = y_test.reshape(1, 10000)

def init_parameters():
    hidden_size = 128
    w1 = np.random.randn(hidden_size, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(10, hidden_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2


def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)


def relu(z):
    return np.maximum(0, z)

def forward_propagation(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def one_hot_encode(y):
    return np.eye(10)[y[0]].T


def relu_derivative(z):
    return z > 0

def backward_propagation(w1, z1, a1, a2, w2, x, y, lambda_reg=0.01):
    m = y.shape[1]
    y_one_hot = one_hot_encode(y)

    dz2 = a2 - y_one_hot
    dw2 = (1 / m) * np.dot(dz2, a1.T) + (lambda_reg / m) * w2
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * relu_derivative(z1)
    dw1 = (1 / m) * np.dot(dz1, x.T) + (lambda_reg / m) * w1
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2


def create_mini_batches(x, y, batch_size):
    m = x.shape[1]
    permutation = np.random.permutation(m)
    x_shuffled = x[:, permutation]
    y_shuffled = y[:, permutation]
    batches = []
    for i in range(0, m, batch_size):
        x_batch = x_shuffled[:, i:i + batch_size]
        y_batch = y_shuffled[:, i:i + batch_size]
        batches.append((x_batch, y_batch))
    return batches


def train_model(x_train, y_train, epochs=50, learning_rate=0.1, batch_size=128, lambda_reg=0.01):
    w1, b1, w2, b2 = init_parameters()
    for epoch in range(epochs):
        batches = create_mini_batches(x_train, y_train, batch_size)
        for x_batch, y_batch in batches:
            z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x_batch)
            dw1, db1, dw2, db2 = backward_propagation(w1, z1, a1, a2, w2, x_batch, y_batch, lambda_reg)
            w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
    return w1, b1, w2, b2

if __name__ == "__main__":

    w1, b1, w2, b2 = train_model(x_train, y_train, epochs=50, learning_rate=0.1, batch_size=128)

    _, _, _, a2_test = forward_propagation(w1, b1, w2, b2, x_test)
    test_predictions = np.argmax(a2_test, axis=0)
    test_accuracy = np.mean(test_predictions == y_test[0])
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    random_indices = np.random.choice(x_test.shape[1], 10, replace=False)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[:, idx].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {test_predictions[idx]}, True: {y_test[0][idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()