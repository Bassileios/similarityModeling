import moviepy.editor
from pydub import AudioSegment
import librosa as lr
import numpy as np

# GLOBAL VARIABLES
vid = ['Muppets-03-04-03','Muppets-02-01-01','Muppets-02-04-04']
path = ['Muppets-03-04-03-sound/','Muppets-02-01-01-sound/','Muppets-02-04-04-sound/']
target = ['Muppets-03-04-03-sound.csv','Muppets-02-01-01-sound.csv','Muppets-02-04-04-sound.csv']
num_frames = [38494,38677,38702]


# Extracts the Sound from a Movie File
def extractor():
	print('Extracting Sound from .avi file')
	for v in vid:
		audio = moviepy.editor.MovieFileClip(v + '.mp3').audio
		audio.write_audiofile(v + '.mp3')
	return
		

# Splits the MP3 File into different overlapping Frames of Sound
def make_sound_frames(num):
	print('Creating the Sound Frames')
	sound = AudioSegment.from_mp3(vid[num])
	# 1 sec = 25 frames --> 1 frame = 1/25 sec
	# 1 sec = 1000 msec
	n = len(sound)
	frame_len = int(1000/25)
	BATCH_SIZE = 5
	count = 0
	for i in range(0,n-frame_len*BATCH_SIZE,frame_len):
		end = i+frame_len*BATCH_SIZE
		part = sound[i:end]
		name = 'frame'+str(count)+'.mp3'
		part.export(path[num]+name, format='mp3')
		count += 1
	return


# Writes a line to the given Database
def add_line(file_name, line):
	with open(file_name, 'a') as file:
		string = ''
		for i in range(len(line)):
			string += str(line[i])
			if i != len(line)-1: string += '\t'
		string += '\n'
		file.write(string)
	return


# Creates the Sound Database
# First the Frame gets loaded
# Then the Hamming Window Function is applied
# Then we create the MEL Spectrogram
# At last it gets transformed with MFCC
def create_features(num):
	print('Strating to create the Sound Database')
	n = num_frames[num]
	nfft = 256
	inv = 1.0/nfft
	sr = 22050

	for i in range(n):
		frame_name = path[num] + 'frame' + str(i+1000) + '.mp3'
		frame, sr = lr.load(frame_name) # = tuple(array, sample rate)

		# Apply Hamming Window
		frame *= np.hamming(len(frame))

		# Fast Fourier Transformation
		magnitude = np.absolute(np.fft.rfft(frame, nfft))
		power =  inv * (magnitude ** 2)

		# Create the MEL Filterbank and use it on the Spectrum
		mel = lr.feature.melspectrogram(S=power, sr=sr, n_fft=int(nfft/2), n_mels=40)

		# Apply MFCC
		final = lr.feature.mfcc(S=mel, sr=sr, n_mfcc=20)
		add_line(target[num], final)
	return



# Adds the sound features to the picture Database to create the final Database of all values measured
def add_sound():
	print('Adding the Sound Features to the Database')

	result = ['Muppets-03-04-03-final.csv', 'Muppets-02-01-01-final.csv', 'Muppets-02-04-04-final.csv']
	databases = ['Muppets-03-04-03-video.csv', 'Muppets-02-01-01-video.csv', 'Muppets-02-04-04-video.csv']
	sound = ['Muppets-03-04-03-sound.csv', 'Muppets-02-01-01-sound.csv', 'Muppets-02-04-04-sound.csv']
	db = ['Muppets-03-04-03_pig.txt','Muppets-02-01-01_pig.txt','Muppets-02-04-04_pig.txt']
	
	for d in range(3):
		X = open(databases[d],'r')
		S = open(sound[d],'r')
		y = open(db[d],'r')

		line_x = X.readline().strip().split()
		line_s = S.readline().strip().split()
		line_y = y.readline().strip().split()
		while len(line_x) > 1 and len(line_s) > 1 and len(line_y) > 1:
			line = line_x[0:len(line_x)-1] + line_s
			line.append(line_y[1])
			add_line(result[d], line)

			line_x = X.readline().strip().split()
			line_s = S.readline().strip().split()
			line_y = y.readline().strip().split()
	return


# Main Procedure, calls the methods in the right order
def main():
	extractor()
	for i in range(3):
		make_sound_frames(i)
		create_features(i)
	add_sound()
	return

