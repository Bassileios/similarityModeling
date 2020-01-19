from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import librosa as lr
import numpy as np

# GLOBAL VARIABLES
sound_vid = ['video1', 'video2', 'video3']
sound_path = ['video1-sound/', 'video2-sound/', 'video3-sound/']
sound_num_frames = [38494, 38677, 38702]
DATA_DIR = '../../data/'


# Extracts the Sound from a Movie File
def extractor():
    print('Extracting Sound from .avi file')
    for v in sound_vid:
        audio = VideoFileClip(DATA_DIR + v + '.avi').audio
        audio.write_audiofile(DATA_DIR + v + '.wav')
    return


# Splits the MP3 File into different overlapping Frames of Sound
def make_sound_frames(num):
    print('Creating the Sound Frames')
    sound = AudioSegment.from_mp3(DATA_DIR + sound_vid[num] + '.mp3')
    # 1 sec = 25 frames --> 1 frame = 1/25 sec
    # 1 sec = 1000 msec
    n = len(sound)
    frame_len = int(1000 / 5)
    BATCH_SIZE = 5
    count = 0
    for i in range(0, n - frame_len * BATCH_SIZE, frame_len):
        end = i + frame_len * BATCH_SIZE
        part = sound[i:end]
        name = sound_vid[num] +'-frame' + str(count//5) + '-' + str(count % 5) + '.wav'
        part.export(DATA_DIR + sound_path[num] + name, format='wav')
        count += 1
    return


if __name__ == '__main__':
    # extractor()
    for i in range(3):
        make_sound_frames(i)