# Print frames from video
import cv2

def create_frames(vidcap,folder):
	success,image = vidcap.read()
	count = 0
	success = True
	while success:
		cv2.imwrite(folder+"/frame%d.jpg" % count, image)     # save frame as JPEG file
		success,image = vidcap.read()
		print 'Read a new frame: ', success
		count += 1

# Main Procedure
print(cv2.__version__)

vid_names = ['Muppets-03-04-03.avi','Muppets-02-04-04.avi','Muppets-02-01-01.avi']
videos = [0]*len(vid_names)
for i in range(len(vid_names)):
	videos[i] = cv2.VideoCapture(vid_names[i])

