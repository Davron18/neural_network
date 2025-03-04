import numpy as np
import matplotlib.pyplot as plt

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

x_train = x_train.reshape(60000, 784).T/255.0
x_test = x_test.reshape(10000, 784).T/255.0

y_train = y_train.reshape(1, 60000)
y_test = y_test.reshape(1, 10000)

def init_parameters():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return w1, b1, w2, b2

def softmax(g):
    return np.exp(g) / np.sum(np.exp(g), axis=0, keepdims=True)

def forward_path(w1, b1, w2, b2, x):
    g1 = np.dot(w1, x) + b1
    y1 = np.maximum(0, g1)
    g2 = np.dot(w2, y1) + b2
    y2 = softmax(g2)
    return g1,y1,g2,y2

def one_hot_embadding(T):
    return np.eye(10)[T[0]].T

def derivative_relu(g):
    return g > 0

def backpropagation(g1, y1, y2, w2, x, T):
    m = T.shape[1]
    one_hot_T = one_hot_embadding(T)
    dz2 = y2 - one_hot_T
    dw2 = 1/m * np.dot(dz2, y1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * derivative_relu(g1)
    dw1 = 1/m * np.dot(dz1, x.T)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2

def update_rule(w1, b1, w2, b2, dw1, db1, dw2, db2, eta):
    w1 = w1 - eta * dw1
    b1 = b1 - eta * db1
    w2 = w2 - eta * dw2
    b2 = b2 - eta * db2
    return w1, b1, w2, b2

def gradient_descent(x, T, iterations, eta):
    w1, b1, w2, b2 = init_parameters()
    for i in range(iterations):
        g1, y1, g2, y2 = forward_path(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backpropagation(g1, y1, y2, w2, x, T)
        w1, b1, w2, b2 = update_rule(w1, b1, w2, b2, dw1, db1, dw2, db2, eta)
    return w1, b1, w2, b2


