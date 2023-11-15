import numpy as np

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    
    def _predict_single(self, x):
        # distances between x and all examples in the training set
        distances = [self.euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # getting indices of the k nearest neighbours
        k_indices = np.argsort(distances)[:self.k]
        
        # getting labels of the k nearest neighbours
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # return the most common class label among the k nearest neighbours
        most_common = np.bincount(k_nearest_labels).argmax()
        
        return most_common
        