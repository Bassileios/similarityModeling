from helpers.generateFrames import generate_all_frames
import os
from datetime import datetime


def MSG(txt):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))


def main():
    if not os.path.exists('../data/video1'):
        MSG("No frames found, extracting frames from videos")
        generate_all_frames()

if __name__ == '__main__':
    main()
