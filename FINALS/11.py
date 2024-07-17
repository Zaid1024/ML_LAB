import numpy as np

def activation(x):
    return 1 if x >= 0 else 0


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])  

w_and = np.array([0.5, 0.5])
b_and = -0.7
w_or = np.array([0.5, 0.5])
b_or = -0.3

lr = 0.1

epochs = 10000
for epoch in range(epochs):
    for x, y_and_true, y_or_true in zip(X, y_and, y_or):
        z_and = np.dot(x, w_and) + b_and
        y_and_pred = activation(z_and)
        z_or = np.dot(x, w_or) + b_or
        y_or_pred = activation(z_or)

        error_and = y_and_true - y_and_pred
        error_or = y_or_true - y_or_pred

        w_and += lr * error_and * x
        b_and += lr * error_and
        w_or += lr * error_or * x
        b_or += lr * error_or

print("AND function:")
for x in X:
    z_and = np.dot(x, w_and) + b_and
    y_and_pred = activation(z_and)
    print(f"Input: {x}, Output: {y_and_pred}")

print("\nOR function:")
for x in X:
    z_or = np.dot(x, w_or) + b_or
    y_or_pred = activation(z_or)
    print(f"Input: {x}, Output: {y_or_pred}")
