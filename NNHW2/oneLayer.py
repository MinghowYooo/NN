import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Binary cross-entropy loss
def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    loss = -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return loss

# Initialize parameters
def initialize_parameters(input_dim, output_dim):
    W = np.random.randn(output_dim, input_dim) * 0.01
    b = np.zeros((output_dim, 1))
    return W, b

# Forward propagation
def forward_propagation(X, W, b):
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    return A, Z

# Backward propagation
def backward_propagation(X, Y, A, Z):
    m = X.shape[1]
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return dW, db

# Training the neural network
def train_nn(X, Y, epochs, learning_rate):
    input_dim = X.shape[0]
    output_dim = Y.shape[0]
    W, b = initialize_parameters(input_dim, output_dim)
    errors = []

    for epoch in range(epochs):
        A, Z = forward_propagation(X, W, b)
        loss = compute_loss(Y, A)
        errors.append(loss)
        dW, db = backward_propagation(X, Y, A, Z)
        W -= learning_rate * dW
        b -= learning_rate * db

        if epoch in [2, 9, 99]:
            plot_decision_boundary(X, Y, W, b, epoch + 1)

    return W, b, errors

# Plot decision boundary
def plot_decision_boundary(X, Y, W, b, epoch):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = sigmoid(np.dot(W, np.c_[xx.ravel(), yy.ravel()].T) + b)
    Z = np.argmax(Z, axis=0).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=0), edgecolors='k', marker='o')
    plt.title(f'Decision Boundary after {epoch} epochs')
    plt.show()

# Main function
if __name__ == "__main__":
    X = np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
                  [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]])
    Y = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    epochs = 100
    learning_rate = 0.1

    W, b, errors = train_nn(X, Y, epochs, learning_rate)

    # Plot training error vs epoch number
    plt.plot(range(epochs), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title('Training Error vs Epoch Number')
    plt.show()