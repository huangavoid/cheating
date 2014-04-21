import numpy as np


class Distance(object):
    def __call__(self, p, q):
        raise NotImplementedError('Must implement method: __call__')


class EuclideanDistance(Distance):
    '''
        欧氏距离
    '''
    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))


class CosineDistance(Distance):
    '''
        余弦距离
    '''
    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return -np.dot(p.T,q) / np.sqrt(np.dot(p,p.T) * np.dot(q,q.T))