import numpy as np


class Feature(object):
    def compute(self, X, y):
        raise NotImplementedError('Must implement method: compute')
    def extract(self, X):
        raise NotImplementedError('Must implement method: extract')


from utils import asColumnMatrix
from operators import ChainOperator


class PCA(Feature):
    '''
        主要成分分析
    '''
    def __init__(self, num_components=0):
        Feature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        if self._num_components<=0 or self._num_components>(XC.shape[1]-1):
            self._num_components = XC.shape[1]-1

        self._mean = XC.mean(axis=1).reshape(-1,1)
        XC = XC-self._mean

        self._eigenvectors, self._eigenvalues, variances = np.linalg.svd(XC, full_matrices=False)

        idx = np.argsort(-self._eigenvalues)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        self._eigenvectors = self._eigenvectors[0:,0:self._num_components].copy()
        self._eigenvalues = self._eigenvalues[0:self._num_components].copy()

        self._eigenvalues = np.power(self._num_components,2) / XC.shape[1]

        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def extract(self, X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self, X):
        X = X - self._mean
        return np.dot(self._eigenvectors.T,X)

    def reconstruct(self, X):
        X = np.dot(self._eigenvectors,X)
        return X + self._mean

    @property
    def num_components(self):
        return self._num_components
    @property
    def eigenvalues(self):
        return self._eigenvalues
    @property
    def eigenvectors(self):
        return self._eigenvectors


class LDA(Feature):
    '''
        线性辨别分析
    '''
    def __init__(self, num_components=0):
        Feature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        XC = asColumnMatrix(X)
        y = np.asarray(y)
        d = XC.shape[0]
        c = len(np.unique(y))
        if self._num_components<=0 or self._num_components>(c-1):
            self._num_components = c-1

        meanTotal = XC.mean(axis=1).reshape(-1,1)
        
        Sw = np.zeros((d,d), dtype=np.float32)
        Sb = np.zeros((d,d), dtype=np.float32)
        for i in range(0,c):
            Xi = XC[:, np.where(y==i)[0]]
            meanClass = np.mean(Xi, axis=1).reshape(-1,1)
            Sw = Sw + np.dot((Xi-meanClass),(Xi-meanClass).T)
            Sb = Sb + Xi.shape[1] * np.dot((meanClass-meanTotal),(meanClass-meanTotal).T)

        self._eigenvalues, self._eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)

        idx = np.argsort(-self._eigenvalues.real)
        self._eigenvalues, self._eigenvectors = self._eigenvalues[idx], self._eigenvectors[:,idx]
        self._eigenvalues = np.array(self._eigenvalues[0:self._num_components].real, dtype=np.float32, copy=True)
        self._eigenvectors = np.array(self._eigenvectors[0:,0:self._num_components].real, dtype=np.float32, copy=True)

        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def project(self, X):
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)

    @property
    def num_components(self):
        return self._num_components
    @property
    def eigenvalues(self):
        return self._eigenvalues
    @property
    def eigenvectors(self):
        return self._eigenvectors


class Fisherfaces(Feature):
    def __init__(self, num_components=0):
        Feature.__init__(self)
        self._num_components = num_components

    def compute(self, X, y):
        XC = asColumnMatrix(X)
        y = np.asarray(y)

        pca = PCA(num_components=(len(y) - len(np.unique(y))))
        lda = LDA(num_components=self._num_components)

        model = ChainOperator(pca, lda)
        model.compute(X, y)
        
        self._eigenvalues = lda.eigenvalues
        self._num_components = lda.num_components
        self._eigenvectors = np.dot(pca.eigenvectors, lda.eigenvectors)

        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def extract(self, X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self, X):
        return np.dot(self._eigenvectors.T, X)

    def reconstruct(self, X):
        return np.dot(self._eigenvectors, X)
        
    @property
    def num_components(self):
        return self._num_components
    @property
    def eigenvalues(self):
        return self._eigenvalues
    @property
    def eigenvectors(self):
        return self._eigenvectors
