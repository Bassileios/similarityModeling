

# Plan
# 1. Daten
# 2. OpenCV herunterladen
# 3. Techniken
	# Farbe: OpenCV -> Color Histogram
	# Textur: OpenCV -> Filter (?)
	# Stimme: ???
	# Bewegungen: ???
	# Umriss: 
	# Augen: Weiß? Form der Pupillen
# 4. Erzeuge aus Video Frames
	# Aus Gruppen von 3 Frames [x-2:x]
	# Anzahl grüner Pixel
	# Anzahl gefundener Muster
	# Bewegungsdifferential
	# PCA der Stimme
# 5. Train Model


import cv2
import pandas as pd
import numpy as np
import mahotas as mt
from PIL import Image

def find_flesh(img):
	im = np.array(img.convert('RGB')) #np.array(Image.open("p.png").convert('RGB'))
	sought = [[178,236,93],[124,252,0],[102,255,0],[172,225,175],[119,221,119],[147,197,114],[133,187,101],[135,169,107],[3,192,60],[120,134,107],[19,136,8],[0,128,0],[85,107,47],[65,72,51]] # generic green hues
	for i in range(len(sought)):
		result += np.count_nonzero(np.all(im==sought[i],axis=2))
	return result


def find_texture(img):
	text = mt.features.haralick(img)
	mean = text.mean(axis=0)
	return mean


def prepare(file_path, prefix):
	with open(file_path, 'r') as file:
		n = 6 #sum(1 for line in file)
		data = [('',0)]*n
		line = file.readline().strip().split()
		for i in range(n):
			data[i] = (prefix.append(line[0]), int(line[1]))
			line = file.readline().strip().split()
		return data
		


def main():
	# Data = DB with picture names and labels
	data = prepare('m_03_04_03_pig_labels', 'Muppets-03-04-03/') # data is now an array of pictures with labels

	BATCH_SIZE = 5 # Number of frames that are looked at simulateneously

	# Optical Flow parameters
	# taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
	lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

	n = 6#len(data)	

	# Start with creation of database
	for i in range(n-BATCH_SIZE):
		# Next we want to work with the data
		flesh = [0]*BATCH_SIZE
		text = [0]*BATCH_SIZE
		motion = [0]*BATCH_SIZE
		old = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
		tracking = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

		# Each frame also calculates the next BATCH_SIZE frames as well
		for j in range(BATCH_SIZE):
			gray = cv2.cvtColor(data[i+j], cv2.COLOR_BGR2GRAY)
			# - Find number of green pixels
			flesh[j] = find_flesh(data[i+j])
			# - Find Textur occurrances in pictures
			text[j] = find_texture(gray)
			# - Find Motion Flow
			motion[j], st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, tracking, None, **lk_params)
			old = gray.copy()






