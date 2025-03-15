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
            x_class = x[y == this_class]
            # Convert data dimention to (channels, epoch * data_point)
            _, n_channels, _ = x_class.shape
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            cov = np.dot(x_class, x_class.T)
            covs.append(cov)

        # Solve generalized eigenvalue problem
        eig_vals, eig_vecs = scipy.linalg.eigh(covs[0], covs[1])
        for i in range(len(eig_vecs)):
            eig_vecs[i] = eig_vecs[i] / np.linalg.norm(eig_vecs[i])

        i = np.argsort(eig_vals)
        ix = np.empty_like(i)
        ix[1::2] = i[: len(i) // 2]
        ix[0::2] = i[len(i) // 2 :][::-1]

        eig_vecs = eig_vecs[:, ix]
        self.filters = eig_vecs.T
        self.patterns = np.linalg.inv(eig_vecs)
        pick_filters = self.filters[: self.n_components]
        x = np.asarray([np.dot(pick_filters, epoch) for epoch in x])
        x = (x**2).mean(axis=2)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x):
        pick_filters = self.filters[: self.n_components]
        x = np.asarray([np.dot(pick_filters, epoch) for epoch in x])
        x = (x**2).mean(axis=2)
        x -= self.mean
        x /= self.std

        return x

    def fit_transform(self, x, y):
        self.fit(x, y)

        return self.transform(x)


            
            