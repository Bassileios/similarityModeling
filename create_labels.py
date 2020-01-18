# Reads in the given File of Time Intervals and Labels
def parse(file_path):
	with open(file_path, 'r') as file:
		stamps = []
		line = file.readline().strip().split()
		i = 0
		while (len(line) >= 3):
			time = line[0]
			time = time.split(':')
			time = int(time[0])*60 + int(time[1])
			frame = time * 25
			stamps.append((frame, int(line[2])))
			i += 1

			line = file.readline().strip().split()
	return stamps





# Creates the Database for the Frames and Labels
def main():
	print('Creating Labels')
	target = ['Muppets-02-01-01','Muppets-02-04-04','Muppets-03-04-03']
	origin = ['m_02_01_01_pig_labels','m_02_04_04_pig_labels','m_03_04_03_pig_labels']
	frames = [38682,38707,38499]

	# Frames are called video-name/frame%d.jpg
	# Labels are written in time stamp intervals
	# Movies are made with 25 frames per second

	for num in range(3):
		bound = parse(origin[num])
	
		doc = open(target[num]+'_pig.txt','w')
	
		label = 0
		count = 0
		for i in range(frames[num]):
			if count < len(bound):
				if bound[count][0] == i:
					label = bound[count][1]
					count += 1
			doc.write("frame{}\t{}\n".format(i,label))

		doc.close()
	return
