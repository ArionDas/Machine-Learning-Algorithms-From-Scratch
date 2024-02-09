import numpy as np    

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y, y_pred):
        m = len(y)
        return (-1/m) * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
    
    def fit(self, x, y):
        m, n = x.shape 
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.epochs):
            y_pred = self.sigmoid(np.dot(x, self.weights) + self.bias)
            gradient = np.dot(x.T, (y_pred - y)) / m     # .T gives the transpose
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.sum(y_pred-y) / m
            
            cost = self.compute_cost(y, y_pred)
            if i%1000 == 0:
                print(f"Cost at iteration {i}", cost)
                
    def predict(self, x):
        y_pred = self.sigmoid(np.dot(x, self.weights) + self.bias)
        return (y_pred > 0.5).astype(int)