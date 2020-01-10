'''
extract the frames form the video file
'''
import cv2
from pathlib import Path


def generate_frames_from_video(filename):
    count = 0
    videofile = filename + ".avi"
    Path(filename).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(videofile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    while cap.isOpened():
        frameId = cap.get(1)  # current frame
        # number
        ret, frame = cap.read()
        if ret is not True:
            break
        if frameId % frameRate//4 == 0:
            imagename = "frame%d.jpg" % count
            save_dir = filename + "/" + imagename
            count += 1
            cv2.imwrite(save_dir, frame)
    cap.release()
    print("Done!")


def generate_all_frames():
    vid_names = ['video1', 'video2', 'video3']
    data_dir = '../data/'
    for video in vid_names:
        generate_frames_from_video(data_dir + video)
