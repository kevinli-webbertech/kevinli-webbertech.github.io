## Deep Learning Network



`import numpy as np`

# Sigmoid activation function and its derivative

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
```

# Neural Network class

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with mean 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # Initialize biases
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

    def feedforward(self, X):
        # Calculate the hidden layer activations
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Calculate the output layer activations
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backpropagate(self, X, y, learning_rate=0.1):
        # Compute the error at the output layer
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        # Compute the error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update the weights and biases
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y, learning_rate)

# Example usage
if __name__ == "__main__":
    # Sample training data (XOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the neural network
    output = nn.feedforward(X)
    print("Predicted Output:")
    print(output)
```

To implement a simple deep learning neural network in Python using the sigmoid activation function, you can use libraries such as `numpy` for the mathematical operations. Below is a basic implementation of a feedforward neural network with one hidden layer:

```python
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with mean 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # Initialize biases
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

    def feedforward(self, X):
        # Calculate the hidden layer activations
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Calculate the output layer activations
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backpropagate(self, X, y, learning_rate=0.1):
        # Compute the error at the output layer
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        # Compute the error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update the weights and biases
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y, learning_rate)

# Example usage
if __name__ == "__main__":
    # Sample training data (XOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the neural network
    output = nn.feedforward(X)
    print("Predicted Output:")
    print(output)
```

### Breakdown of the Code

1. **Sigmoid Activation Function**: The `sigmoid` function maps inputs to values between 0 and 1, which is useful for binary classification problems. The derivative of the sigmoid function is needed for backpropagation.

2. **Neural Network Class**:
   - The class `NeuralNetwork` contains methods for initializing the network, performing forward propagation (`feedforward`), and updating the weights using backpropagation (`backpropagate`).

3. **Feedforward Process**:
   - The input data is passed through the network, which computes activations for the hidden and output layers.

4. **Backpropagation Process**:
   - After computing the output, the error is propagated back to adjust the weights and biases using the derivative of the sigmoid function.

5. **Training**:
   - The `train` method runs the backpropagation process for a specified number of epochs, updating the weights after each iteration.

6. **Example**:
   - The example in the `if __name__ == "__main__":` block demonstrates how to train the neural network to solve a simple XOR problem.

### Notes

- This is a basic implementation of a neural network. In practice, you would use libraries like TensorFlow or PyTorch for large-scale and more efficient training of neural networks.
