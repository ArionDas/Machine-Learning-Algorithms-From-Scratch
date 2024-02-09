import numpy as np  

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, x):
        self.hidden_output = self.sigmoid(np.dot(x, self.w1) + self.b1)
        
        self.output = self.sigmoid(np.dot(self.hidden_output, self.w2) + self.b2)
        return self.output
    
    def backward(self, x, y, learning_rate):
        d_output = (y - self.output) * self.sigmoid_derivative(self.output)
        d_w2 = np.dot(self.hidden_output.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)
        
        d_hidden = np.dot(d_output, self.w2.T) * self.sigmoid_derivative(self.hidden_output)
        d_w1 = np.dot(x.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        self.w2 += learning_rate * d_w2
        self.b2 += learning_rate * d_b2
        self.w1 += learning_rate * d_w1
        self.b1 += learning_rate * d_b1
        
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, learning_rate)
            loss = np.mean((y-output)**2)
            
    def predict(self, x):
        return self.forward(x)