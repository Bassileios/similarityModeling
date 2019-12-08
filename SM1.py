

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


import opencv-python as cv
import pandas as pd
import numpy as np
from PIL import Image

def find_green(img):
	im = np.array(img.convert('RGB')) #np.array(Image.open("p.png").convert('RGB'))
	sought = [[178,236,93],[124,252,0],[102,255,0],[172,225,175],[119,221,119],[147,197,114],[133,187,101],[135,169,107],[3,192,60],[120,134,107],[19,136,8],[0,128,0],[85,107,47],[65,72,51]] # generic green hues
	for i in range(len(sought)):
		result += np.count_nonzero(np.all(im==sought[i],axis=2))
	return result

# 1. Load in Data
# Data = DB with picture names and labels
data = open('material/Muppets_03_04_03_kermit.txt','r')
# data is now an array of pictures with labels
# This needs to get transformed to a matrix of n x (k+1) where k is the batch size
BATCH_SIZE = 5
n = len(data)[0]
data = prepare(data, BATCH_SIZE) # may be included in previous step

# Start with creation of database
for i in range(n):
	# Next we want to work with the data
	green = [0]*k
	for j in range(BATCH_SIZE):
		# - Find number of green pixels
		green[j] = find_green(data[i][j])
		# - Find Textur occurrances in pictures
		text[j] = find_texture(data[i][j])
		# - Find Motion Flow
		motion[j] = cv.optical_flow(data[i][j])





