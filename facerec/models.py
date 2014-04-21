from features import Feature
from classifiers import Classifier
from features import Fisherfaces
from classifiers import NearestNeighbor


class PredictableModel(object):
    def __init__(self, feature, classifier):
        if not isinstance(feature, Feature):
            raise TypeError('must be type of Feature')
        if not isinstance(classifier, Classifier):
            raise TypeError('must be type of Classifier')
        self.feature = feature
        self.classifier = classifier

    def compute(self, X, y):
        features = self.feature.compute(X, y)
        self.classifier.compute(features, y)

    def predict(self, X):
        q = self.feature.extract(X)
        return self.classifier.predict(q)

class ExtendedPredictableModel(PredictableModel):
	def __init__(self, sub_dirnames):
		PredictableModel.__init__(self, feature=Fisherfaces(), classifier=NearestNeighbor())
		#self.face_sz = face_sz
		self.sub_dirnames = sub_dirnames
