import os
import cv2
import numpy as np


def asRowMatrix(X):
	if len(X)==0:
		return np.array([])
	total = 1
	for i in range(0, np.ndim(X[0])):
		total = total * X[0].shape[i]
	mat = np.empty([0, total], dtype=X[0].dtype)
	for row in X:
		mat = np.append(mat, row.reshape(1,-1), axis=0)
	return np.asmatrix(mat)

def asColumnMatrix(X):
	if len(X)==0:
		return np.array([])
	total = 1
	for i in range(0, np.ndim(X[0])):
		total = total * X[0].shape[i]
	mat = np.empty([total, 0], dtype=X[0].dtype)
	for col in X:
		mat = np.append(mat, col.reshape(-1,1), axis=1)
	return np.asmatrix(mat)

def draw_rect(dst, (x0,y0), (x1,y1)):
	cv2.rectangle(dst, (x0,y0), (x1,y1), (0,255,0), 1)

def draw_text(dst, (x,y), text):
	cv2.putText(dst, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=1, lineType=cv2.CV_AA)

def read_images(path, imsize=None):
	X, y, foldernames = [], [], []
	c = 0
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			foldernames.append(subdirname)
			subpath = os.path.join(dirname, subdirname)
			for filename in os.listdir(subpath):
				try:
					im = cv2.imread(os.path.join(subpath, filename), cv2.IMREAD_GRAYSCALE)
					if imsize is not None:
						im = cv2.resize(im, imsize)
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(c)
				except:
					print 'I/O Error'
					raise
			c = c + 1
	return [X, y, foldernames]

def plot_images(images, rows, cols):
	fig = plt.figure()
	for i in xrange(len(images)):
		ax = fig.add_subplot(rows, cols, (i+1))
		plt.setp(ax.get_xticklabels(), visible=True)
		plt.setp(ax.get_yticklabels(), visible=True)
		plt.imshow(np.asarray(images[i]))
	plt.show()

####################

import cPickle

def save_model(filename, model):
	output = open(filename, 'wb')
	cPickle.dump(model, output)
	output.close()
    
def load_model(filename):
	pkl_file = open(filename, 'rb')
	res = cPickle.load(pkl_file)
	pkl_file.close()
	return res
