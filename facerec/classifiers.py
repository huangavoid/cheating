import numpy as np
import operator as op
from distances import EuclideanDistance
from utils import asRowMatrix


class Classifier(object):
    def compute(self, X, y):
        raise NotImplementedError('Must implement method: compute')

    def predict(self, X):
        raise NotImplementedError('Must implement method: predict')


class NearestNeighbor(Classifier):
    '''
        k近邻
    '''
    def __init__(self, dist_matric=EuclideanDistance(), k=1):
        Classifier.__init__(self)
        self.k = k
        self.dist_matric = dist_matric

    def compute(self, X, y):
        self.X = X
        self.y = np.asarray(y)

    def predict(self, q):
        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_matric(xi, q)
            distances.append(d)
        if len(distances) > len(self.y):
            raise Exception('Distance Matric Incorrect')
        distances = np.asarray(distances)
        
        idx = np.argsort(distances)
        sorted_y = self.y[idx]
        sorted_distances = distances[idx]

        sorted_y = sorted_y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]

        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
        predicted_label = max(hist.iteritems(), key=op.itemgetter(1))[0]

        return [predicted_label, { 'labels': sorted_y, 'distances':sorted_distances }]