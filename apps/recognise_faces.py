import sys, os
import cv2
import numpy as np
sys.path.append('..')
from facerec.features import Fisherfaces
from facerec.classifiers import NearestNeighbor
from facerec.models import PredictableModel
from facedet.detectors import SkinFaceDetector, CascadedDetector
from facerec.utils import read_images, save_model, load_model, draw_rect, draw_text

class ExtendedPredictableModel(PredictableModel):
	def __init__(self, names):
		PredictableModel.__init__(self, feature=Fisherfaces(), classifier=NearestNeighbor())
		self.names = names

def recognise_faces_by_img(train_dir, test_dir, face_sz=(100,100)):	
	if os.path.exists('model.pkl'):
		face_model = load_model('model.pkl')
	else:
		[images, labels, foldernames] = read_images(train_dir, face_sz)
		list_of_labels = list(xrange(max(labels)+1))
		sub_dirnames = dict(zip(list_of_labels, foldernames))
			
		face_model = ExtendedPredictableModel(names=sub_dirnames)
		face_model.compute(images, labels)
		save_model('model.pkl', face_model)
	
	expectations, predictions = [], []
	for dirname, dirnames, filenames in os.walk(test_dir):
		for subdirname in dirnames:
			subpath = os.path.join(dirname, subdirname)
			for filename in os.listdir(subpath):
				test = cv2.imread(os.path.join(subpath, filename), cv2.IMREAD_GRAYSCALE)
				test = cv2.resize(test, face_sz, interpolation=cv2.INTER_CUBIC)
				prediction = face_model.predict(test)[0]
				expectations.append(subdirname)
				predictions.append(face_model.names[prediction])
				print subdirname, face_model.names[prediction]

	total = len(expectations)
	rate = 0
	for i in range(total):
		if expectations[i] == predictions[i]:
			rate = rate + 1 
	print rate, total, rate*100.0/total

def recognise_faces_by_cam(train_dir, face_sz=(100,100)):
	if os.path.exists('model.pkl'):
		face_model = load_model('model.pkl')
	else:
		[images, labels, foldernames] = read_images(train_dir, face_sz)
		list_of_labels = list(xrange(max(labels)+1))
		sub_dirnames = dict(zip(list_of_labels, foldernames))
			
		face_model = ExtendedPredictableModel(names=sub_dirnames)
		face_model.compute(images, labels)
		save_model('model.pkl', face_model)
	
	face_detector = SkinFaceDetector()
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		if frame is not None:
			img = frame.copy()
			img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation=cv2.INTER_CUBIC)
			img_tmp = img.copy()
			for i, [x0,y0,x1,y1] in enumerate(face_detector.detect(img)):
				face = img[y0:y1, x0:x1]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, face_sz, interpolation=cv2.INTER_CUBIC)
				prediction = face_model.predict(face)[0]
				
				draw_rect(img_tmp, (x0,y0), (x1,y1))
				draw_text(img_tmp, (x0-20,y0-20), face_model.names[prediction])
			cv2.imshow('recoginise faces by cam', img_tmp)
		if cv2.waitKey(500) & 0xFF == 27:
			break
	cv2.destroyAllWindows()

if __name__ == '__main__':
	#if len(sys.argv) != 3:
	#	print "<USAGE>: python extract_faces.py <train_dir> <test_dir>"
	#	sys.exit()
	#recognise_faces_by_img(train_dir=sys.argv[1], test_dir=sys.argv[2])
	
	if len(sys.argv) != 2:
		print "<USAGE>: python extract_faces.py <train_dir>"
		sys.exit()
	recognise_faces_by_cam(train_dir=sys.argv[1])
