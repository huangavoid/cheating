import numpy as np
from features import Feature


class FeatureOperator(Feature):
    def __init__(self, model1, model2):
        if (not isinstance(model1, Feature)) or (not isinstance(model2, Feature)):
            raise Exception('Must be type of Feature')
        self.model1 = model1
        self.model2 = model2


class ChainOperator(FeatureOperator):
    def __init__(self, model1, model2):
        FeatureOperator.__init__(self, model1, model2)

    def compute(self, X, y):
        X = self.model1.compute(X, y)
        return self.model2.compute(X, y)

    def extract(self, X):
        X = self.model1.extract(X)
        return self.model2.extract(X)