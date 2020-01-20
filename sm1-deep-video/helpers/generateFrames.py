'''
extract the frames form the video file
'''
import cv2
from pathlib import Path


def generate_frames_from_video(filedir,filename):
    count = 0
    savedir = filedir + filename
    videofile = filedir + filename + ".avi"
    Path(savedir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(videofile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    while cap.isOpened():
        frameId = cap.get(1)  # current frame
        # number
        ret, frame = cap.read()
        if ret is not True:
            break
        if frameId % frameRate//5 == 0:
            imagename = filename + "_frame%d.jpg" % count
            savename = savedir + "/" + imagename
            count += 1
            cv2.imwrite(savename, frame)
    cap.release()
    print("Done!")


def generate_all_frames():
    vid_names = ['video1', 'video2', 'video3']
    data_dir = '../data/'
    for video in vid_names:
        generate_frames_from_video(filedir=data_dir, filename=video)
