import numpy as np


class K_Means:
    def __init__(self, k=3, tolerance = 0.001, max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def fit(self, data):
        
        self.centroids = {}
        
        ## randomly initializing the centroids
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        ## begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            
            for i in range(self.k):
                self.classes[i] = []
                
                ## finding the distance between the point and cluster; choosing the nearest centroid
                for features in data:
                    distances = [np.lingalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classes[classification].append(features)
                    
                previous = dict(self.centroids)
                
                ## average the cluster datapoints to re-calculate the centroids\
                for classification in self.classes:
                    self.centroids[classification] = np.average(self.classes[classification], axis=0)
                    
                isOptimal = True
                
                for centroid in self.centroids:
                    original_centroid = previous[centroid]
                    curr = self.centroids[centroid]
                    
                    if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                        isOptimal = False
                        
                if isOptimal:
                    break
                
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification