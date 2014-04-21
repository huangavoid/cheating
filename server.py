import os
import time
import numpy as np
import cv2

import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.websocket

from facedet.detectors import SkinFaceDetector, CascadedDetector
from facerec.models import ExtendedPredictableModel
from facerec.utils import read_images, save_model, load_model

###########

class FaceController(object):
	def __init__(self):
		self.face_sz = (100,100)
		#self.detector = SkinFaceDetector()
		self.detector = CascadedDetector()

	
	def doPreparement(self, uname):
		self.raw_path = 'data/raw/' + uname
		if os.path.exists(self.raw_path):
			self.clearImg(self.raw_path)
		else:
			os.mkdir(self.raw_path)
		
		self.train_path = 'data/train/' + uname
		if os.path.exists(self.train_path):
			self.clearImg(self.train_path)
		else:
			os.mkdir(self.train_path)
	
		self.test_path = 'data/test/' + uname
		if os.path.exists(self.test_path):
			self.clearImg(self.test_path)
		else:
			os.mkdir(self.test_path)

	def saveImg(self, path, img_data):
		item_number = 1
		for item in img_data:
			itempath = "%s/%d.jpg" % (path, item_number)
			print itempath
			cv2.imwrite(itempath, item)
			item_number += 1

	def clearImg(self, path):
		for filename in os.listdir(path):
			os.remove(os.path.join(path,filename))   
		#os.rmdir(path)
		
	def captureFace(self, path):
		face_data = [] 
		for filename in os.listdir(path):
			img = cv2.imread(os.path.join(path,filename))
			rects = self.detector.detect(img)
			for i,rect in enumerate(rects):
				x0,y0,x1,y1 = rect
				face = img[y0:y1,x0:x1]
				face = cv2.resize(face, self.face_sz, interpolation=cv2.INTER_CUBIC)
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				face_data.append(face)
		return face_data
	
	def makeTrainSample(self):
		self.clearImg(self.train_path)
		face_data = self.captureFace(self.raw_path)
		self.saveImg(self.train_path, face_data)
		self.clearImg(self.raw_path)

	def makeTestSample(self):
		self.clearImg(self.test_path)
		face_data = self.captureFace(self.raw_path)
		self.saveImg(self.test_path, face_data)
		self.clearImg(self.raw_path)
	
	def recogniseFace(self):
		train_url = os.path.split(self.train_path)[0]
		tester_name = os.path.split(self.test_path)[1]
		
		if os.path.exists('model.pkl'):
			self.predictModel = load_model('model.pkl')
		else:
			[images, labels, foldernames] = read_images(train_url, self.face_sz)
			list_of_labels = list(xrange(max(labels)+1))
			sub_dirnames = dict(zip(list_of_labels, foldernames))
			
			self.predictModel = ExtendedPredictableModel(sub_dirnames=sub_dirnames)
			self.predictModel.compute(images, labels)
			save_model('model.pkl', self.predictModel)
		
		expectations, predictions = [], []
		for filename in os.listdir(self.test_path):
			test = cv2.imread(os.path.join(self.test_path, filename), cv2.IMREAD_GRAYSCALE)
			test = cv2.resize(test, self.face_sz, interpolation=cv2.INTER_CUBIC)
			prediction = self.predictModel.predict(test)[0]
			expectations.append(tester_name)
			predictions.append(self.predictModel.sub_dirnames[prediction])
			print tester_name, self.predictModel.sub_dirnames[prediction]

		total = len(expectations)
		rate = 0
		for i in range(total):
			if expectations[i] == predictions[i]:
				rate = rate + 1 
		print rate, total, rate*100.0/total
		return rate*100.0/total
	
	def isMultiplayer(self):
		## 1. how many faces
		face_data = self.captureFace(self.raw_path)
		if len(face_data) > 15:
			#self.makeTestSample()
			self.clearImg(self.raw_path)
			self.clearImg(self.test_path)
			return 'YES'
		self.makeTestSample()
		return 'NO'
	
	def isYourself(self):
		## 2. who are you
		if self.recogniseFace() > 50.0:
			return 'YES'
		return 'NO'
		## 3. where do you glance at
	
##########

class IndexHandler(tornado.web.RequestHandler):
	def get(self):
		self.render('index.html')

class InfoHandler(tornado.web.RequestHandler):
	''' get tester name and mkdir '''
	def post(self):
		uname = self.get_argument('username')
		self.application.faceController.doPreparement(uname)
		self.render('info.html')

class GetimgHandler(tornado.web.RequestHandler):
	''' turn on webcam '''
	def get(self):
		next = '/' + self.get_argument('next')
		self.render('getimg.html', next=next)

class WebcamHandler(tornado.websocket.WebSocketHandler):
	''' save img to raw '''
	def open(self):
		pass
	def on_close(self):
		pass
	def on_message(self, message):
		#self.write_message(message)
		arr = np.asarray(bytearray(message), dtype=np.uint8)
		img = cv2.imdecode(arr,-1)
		if img is not None:
			imgpath = "%s/%s.jpg" % (self.application.faceController.raw_path, str(time.time()))
			#imgpath = self.application.faceController.raw_path + '/' + str(time.time()) + '.jpg'
			cv2.imwrite(imgpath, img) 

class Info2Handler(tornado.web.RequestHandler):
	''' make train sample '''
	def get(self):
		self.application.faceController.makeTrainSample()
		self.render('info2.html')

class Info3Handler(tornado.web.RequestHandler):
	''' is multiplayer ? '''
	def get(self):
		result = self.application.faceController.isMultiplayer()
		self.render('info3.html', result=result)

class Info4Handler(tornado.web.RequestHandler):
	''' is yourself ? '''
	def get(self):
		result = self.application.faceController.isYourself()
		self.render('info4.html', result=result)

class EndHandler(tornado.web.RequestHandler):
	def get(self):
		self.render('end.html')


###########

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class Application(tornado.web.Application):
	def __init__(self):
		self.faceController = FaceController()
		handlers = [
			(r'/', IndexHandler),
			(r'/info', InfoHandler),
			(r'/getimg', GetimgHandler),
			(r'/webcam', WebcamHandler),
			(r'/info2', Info2Handler),
			(r'/info3', Info3Handler),
			(r'/info4', Info4Handler),
			(r'/end', EndHandler),
		]
		settings = dict(
			template_path=os.path.join(os.path.dirname(__file__), "templates"),
			static_path=os.path.join(os.path.dirname(__file__), "static"),
			debug=True,
		)
		tornado.web.Application.__init__(self, handlers, **settings)

if __name__=='__main__':
	tornado.options.parse_command_line()
	http_server = tornado.httpserver.HTTPServer(Application())
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()
