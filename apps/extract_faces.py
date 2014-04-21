import os, sys
# append facedet to module search path
sys.path.append("..")
import cv2
import numpy as np
from facedet.detectors import SkinFaceDetector, CascadedDetector


def extract_faces_by_img(src_dir, dst_dir, detector, face_sz = (100,100)):
	"""
	1. Extracts the faces from all images in a given src_dir 
	2. Writes the extracted faces to dst_dir. 
	3. Needs a facedet.Detector object to perform the actual detection.
	"""
	if not os.path.exists(dst_dir):
		try:
			os.mkdir(dst_dir)
		except:
			raise OSError("Can't create destination directory (%s)!" % (dst_dir))
	for dirname, dirnames, filenames in os.walk(src_dir):
		for subdir in dirnames:
				src_subdir = os.path.join(dirname, subdir)
				dst_subdir = os.path.join(dst_dir,subdir)
				if not os.path.exists(dst_subdir):
					try:
						os.mkdir(dst_subdir)
					except:
						raise OSError("Can't create destination directory (%s)!" % (dst_subdir))
				for filename in os.listdir(src_subdir):
					name, ext = os.path.splitext(filename)
					src_fn = os.path.join(src_subdir,filename)
					img = cv2.imread(src_fn)
					rects = detector.detect(img)
					for i,rect in enumerate(rects):
						x0,y0,x1,y1 = rect
						face = img[y0:y1,x0:x1]
						face = cv2.resize(face, face_sz, interpolation = cv2.INTER_CUBIC)
						#face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
						print os.path.join(dst_subdir, "%s_%d%s" % (name,i,ext))
						cv2.imwrite(os.path.join(dst_subdir, "%s_%d%s" % (name,i,ext)), face)

def extract_faces_by_cam(dst_dir, detector, face_count=20, face_sz=(100,100)):
	"""
	1. Extracts the faces from webcam
	2. Writes the extracted faces to dst_dir. 
	3. Needs a facedet.Detector object to perform the actual detection.
	"""
	if not os.path.exists(dst_dir):
		try:
			os.mkdir(dst_dir)
		except:
			raise OSError("Can't create destination directory (%s)!" % (dst_dir))
	
	count = 1
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		if frame is not None:
			img = frame.copy()
			img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation=cv2.INTER_CUBIC)
			img_tmp = img.copy()
			for i,[x0,y0,x1,y1] in enumerate(detector.detect(img)):
				face = img[y0:y1,x0:x1]
				face = cv2.resize(face, face_sz, interpolation = cv2.INTER_CUBIC)
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				print os.path.join(dst_dir, "%d_%d.jpg" % (count,i))
				cv2.imwrite(os.path.join(dst_dir, "%d_%d.jpg" % (count,i)), face)
				count += 1
			cv2.imshow('extract face by cam',img_tmp)
		if count > face_count:
			break
		if cv2.waitKey(500) & 0xFF == 27:
			break
	cv2.destroyAllWindows()

if __name__ == "__main__":
	#if len(sys.argv) != 3:
	#	print "<USAGE>: python extract_faces.py <src_dir> <dst_dir>"
	#	sys.exit()
	#extract_faces_by_img(dst_dir=sys.argv[1], dst_dir=sys.argv[2], detector=CascadedDetector())
	#extract_faces_by_img(src_dir=sys.argv[1], dst_dir=sys.argv[2], detector=SkinFaceDetector())
	
	if len(sys.argv) != 2:
		print "<USAGE>: python extract_faces.py <dst_dir>"
		sys.exit()
	#extract_faces_by_cam(dst_dir=sys.argv[1], detector=CascadedDetector())
	extract_faces_by_cam(dst_dir=sys.argv[1], detector=SkinFaceDetector())

