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





# Creates the Database for the Frames and labels

names = ['Muppets-02-01-01','Muppets-02-04-04','Muppets-03-04-03']
frames = [38682,38707,38499]

# Frames are called video-name/frame%d.jpg
# Labels are written in time stamp intervals
# Movies are made with 25 frames per second

bound = parse('m_03_04_03_kermit_labels') # CHANGE FILE TO LABELS ONCE DONE

doc = open('Muppets_03_04_03_kermit.txt','w') # CHANGE ACCORDINGLY

label = 0
count = 0
for i in range(frames[2]): # CHANGE ACCORDINGLY
	if bound[count][0] == i:
		label = bound[count][1]
	doc.write("frame{}\t{}\n".format(i,label))

doc.close()
