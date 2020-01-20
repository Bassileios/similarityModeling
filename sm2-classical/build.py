import create_sound
import create_labels
import create_frames
import create_database

# Splits the Video up into Pictures
create_frames.main()

# Takes the List of Labels and Timeintervals and converts them into a list of frame and label
create_labels.main()

# Extracts the Features of one Batch of Frames each time, creating the Feature Database
# Expected Runtime: 30h
create_database.main()

# Extracts the Sound from the Videos
# Saves the Sound to Sound-Frames
# Extracts the Sound Features
# Expected Runtime: 10h
create_sound.main()
