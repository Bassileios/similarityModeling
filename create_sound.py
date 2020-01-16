import moviepy.editor
from pydub import AudioSegment
import os

def write_sound(audio, file_name, suffix):
	audio.write_audiofile(file_name + suffix)


def retrieve_sound(file_name, suffix):
	return moviepy.editor.MovieFileClip(file_name + suffix).audio


def extractor():
	vid = ['Muppets-03-04-03','Muppets-02-01-01','Muppets-02-04-04']
	for v in vid:
		audio = retrieve_sound(v, '.mp3')
		write_sound(audio, v, '.mp3')
		

def make_sound_frames(num):
	vid = ['Muppets-03-04-03.mp3','Muppets-02-01-01.mp3','Muppets-02-04-04.mp3']
	path = ['Muppets-03-04-03-sound/','Muppets-02-01-01-sound/','Muppets-02-04-04-sound/']

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


make_sound_frames(1)
make_sound_frames(2)
