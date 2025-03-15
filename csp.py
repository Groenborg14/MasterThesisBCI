import scipy
import numpy as np



class CSP:
    def __init__(self, n_components=8):
       self.n_components = n_components

    def fit(self,x,y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        if n_classes != 2:
            raise ValueError('n_classes must be 2')
        
        covs = []
        for this_class in self.classes:
            X_class = x[y == this_class]
            
            