import numpy as np
import matplotlib.pyplot as plt

# Sample data points
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array([
    [-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321]
])

# Hyperparameters
input_size = 1
hidden_size = 10
output_size = 1
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(hidden_size, input_size)
b1 = np.random.randn(hidden_size, 1)
W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size, 1)

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
errors = []
for epoch in range(epochs):
    # Forward propagation
    X_input = X.reshape(-1, 1)
    Y_input = Y.reshape(-1, 1)
    
    Z1 = np.dot(W1, X_input.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2  # Linear activation for output layer
    
    # Compute loss
    loss = np.mean((A2 - Y_input.T) ** 2)
    errors.append(loss)
    
    # Backward propagation
    dZ2 = A2 - Y_input.T
    dW2 = np.dot(dZ2, A1.T) / X_input.shape[0]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X_input.shape[0]
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(dZ1, X_input) / X_input.shape[0]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X_input.shape[0]
    
    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # Plot results at specified epochs
    if epoch in [10, 100, 200, 400, 1000]:
        plt.figure()
        plt.scatter(X, Y, color='red', label='Actual')
        plt.scatter(X, A2.T, color='blue', label='Predicted')
        plt.title(f'Actual vs Predicted at Epoch {epoch}')
        plt.legend()
        plt.show()

# Plot training error vs epoch number
plt.figure()
plt.plot(errors)
plt.title('Training Error vs Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()