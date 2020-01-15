

# Plan
# 1. Daten
# 2. OpenCV herunterladen
# 3. Techniken
	# Farbe: OpenCV -> Color Histogram
	# Textur: OpenCV -> Filter (?)
	# Stimme: ???
	# Bewegungen: OpenCV -> Optical Flow
	# Umriss: ???
# 4. Erzeuge aus Video Frames
	# Aus Gruppen von 5 Frames [x-4:x]
	# Anzahl rosa Pixel
	# Anzahl gefundener Muster
	# Bewegungsdifferential -> check
	# PCA der Stimme
# 5. Train Model


import cv2
import pandas as pd
import numpy as np
import mahotas as mt
import math
from PIL import Image

def find_flesh(img):
	#im = np.array(img.convert('RGB')) #np.array(Image.open("p.png").convert('RGB'))
	sought = []
	result = 0
	for i in range(len(sought)):
		result += np.count_nonzero(np.all(img==sought[i],axis=2))
	return result


def find_texture(img):
	text = mt.features.haralick(img)
	mean = text.mean(axis=0)
	return mean


def prepare(file_path, prefix, suffix):
	with open(file_path, 'r') as file:
		n = 10 #sum(1 for line in file)
		data = [('',0)]*n
		line = file.readline().strip().split()
		for i in range(16000): line = file.readline().strip().split()
		for i in range(n):
			data[i] = (prefix +line[0]+suffix, int(line[1]))
			line = file.readline().strip().split()
		return data
		


def main():
	# Data = DB with picture names and labels
	data = prepare('Muppets_03_04_03_pig.txt', 'Muppets-03-04-03/', '.jpg') # data is now an array of pictures with labels

	BATCH_SIZE = 5 # Number of frames that are looked at simulateneously
	textures = [2,10,11,12]

	# Optical Flow parameters
	of_params = dict(winSize = (30,20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	# ShiTomasi corner detection
	corner_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

	n = 10#len(data)	

	# Start with creation of database
	for i in range(n-BATCH_SIZE):
		# Next we want to work with the data
		flesh = [0]*BATCH_SIZE
		text = [0]*(BATCH_SIZE*len(textures))
		motion = [0]*BATCH_SIZE
		image = cv2.imread(data[i][0])
		label = data[i+BATCH_SIZE][1]
		old = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		tracking = cv2.goodFeaturesToTrack(old, mask = None, **corner_params)
		tex_count = 0
		# Each frame also calculates the next BATCH_SIZE frames as well
		for j in range(BATCH_SIZE):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# - Find number of flesh pixels
			flesh[j] = find_flesh(image)
			# - Find Textur occurrances in pictures
			texture = find_texture(gray)
			for t in textures:
				text[tex_count] = texture[t]
				tex_count += 1
			
			# - Find Motion Flow
			if isinstance(tracking,np.ndarray):
				points, st, err = cv2.calcOpticalFlowPyrLK(old, gray, tracking, None, **of_params)
				motion[j] = points[st==1]
				old = gray.copy()
				tracking = motion[j].reshape(-1,1,2)
			if j != BATCH_SIZE-1: image = cv2.imread(data[i+j+1][0])

		# Analyze the flow of movement
		movement = [0]*(BATCH_SIZE-1)
		if isinstance(tracking,np.ndarray):
			for m in range(len(motion[0][0])):
				for j in range(BATCH_SIZE-1):
					diff_x = motion[j][m][0] - motion[j+1][m][0]
					diff_y = motion[j][m][1] - motion[j+1][m][1]
					movement[j] = math.sqrt(diff_x*diff_x + diff_y*diff_y)

		print('Flesh:',flesh,'text:',text,'motion:',movement, 'Label:', label)



def analyse_mp():
	pictures = ['miss_piggy_1.jpg','miss_piggy_2.jpg','miss_piggy_3.jpg']
	text = [0,0,0]
	for i in range(3):
		image = cv2.imread(pictures[i])
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		text[i] = find_texture(gray)
	total = [0]*13
	for i in range(13):
		total[i] = math.sqrt((text[0][i]-text[1][i])*(text[0][i]-text[1][i]) + (text[1][i]-text[2][i])*(text[1][i]-text[2][i]) + (text[0][i]-text[2][i])*(text[0][i]-text[2][i]))
		total[i] /= max(text[0][i],text[1][i],text[2][i])
	print(total)
	# Analysis shows that in all three pictures haralick features 2,10,11,12 are almost exactly the same
	print(text[0][2],text[0][10],text[0][11],text[0][12])
	print(text[1][2],text[1][10],text[1][11],text[1][12])
	print(text[2][2],text[2][10],text[2][11],text[2][12])
	

analyse_mp()

#main()



