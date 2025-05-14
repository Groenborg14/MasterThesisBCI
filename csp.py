import scipy
import numpy as np

from sklearn.covariance import LedoitWolf

import numpy as np
import scipy.linalg

class CSP:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.m = n_components // 2

    def fit(self, x, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        if n_classes != 2:
            raise ValueError('n_classes must be 2')
        #print(f"x shape: {x.shape}")
        #print(f"y shape: {y.shape}")
        #print(f"Unique classes in y: {np.unique(y)}")

        covs = []
        for this_class in self.classes:
            idx = np.where(y == this_class)[0]  # Ensuring safe indexing
            x_class = x[idx]

            n_trials, n_channels, _ = x_class.shape
            #x_class = np.transpose(x_class, [1, 0, 2]).reshape(n_channels, -1)

            #cov = np.cov(x_class, rowvar=True, bias=True)
            #cov += np.eye(cov.shape[0]) * 1e-6
            #covs.append(cov)
            lw = LedoitWolf()
            #cov = lw.fit(x_class.T).covariance_  # shape: (n_channels, n_channels)

            c = np.zeros((n_trials, n_channels, n_channels))
            for i in range(n_trials):
                c[i] = lw.fit(x_class[i].T).covariance_
            
            covs.append(c.mean(axis=0))  # shape: (n_channels, n_channels)

        print("cov matrices:",covs[0].shape, covs[1].shape)
        # Solve the generalized eigenvalue problem
        eig_vals, eig_vecs = scipy.linalg.eig(covs[0], covs[1])
        
        # Normalize eigenvectors
        #eig_vecs /= np.linalg.norm(eig_vecs, axis=0)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, sorted_indices]

        self.filters = eig_vecs.T
        self.filters = np.concatenate([self.filters[:self.m], self.filters[-self.m:]], axis=0)

        self.patterns = np.linalg.pinv(eig_vecs)

        # Apply selected filters
       # pick_filters = self.filters[: self.n_components]
        #x_transformed = np.asarray([np.dot(pick_filters, epoch) for epoch in x])
        #x_transformed = (x_transformed**2).mean(axis=2)
        #x_transformed = np.log((x_transformed**2).mean(axis=2))

        #self.mean = x_transformed.mean(axis=0)
        #self.std = x_transformed.std(axis=0)
        #self.std[self.std == 0] = 1  # avoid division by zero
        #x_transformed = (x_transformed - self.mean) / self.std
        

    def transform(self, x):
        pick_filters = self.filters#[: self.n_components]
        #print(pick_filters.shape)
        x_transformed = np.asarray([np.dot(pick_filters, epoch) for epoch in x])
        x_transformed = np.log(x_transformed**2).mean(axis=2)

        # ➕ Apply the same Z-score normalization
       # x_transformed = (x_transformed - self.mean) / self.std

        return x_transformed

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


            
            