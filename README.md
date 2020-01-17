# SimilarityModeling
Implementation of Similarity Modeling Project 2
## Problem
- Detection of the pigs in the given video files

## Approach
In this project classical feature enigneering approach was used for detection of the pigs.

## Libraries used
- Numpy
- Mahotas
- Sci-kit Learn
- OpenCV
- Librosa
- Moviepy.editor
- Pydub

## USAGE
This Project is split into
- create_frames.py: Extracts the frames from the video data
- create_labels.py: Based on the label and timestamp lists (m_02_01_01_pig_labels, m_02_04_04_pig_labels, m_03_04_03_pig_labels) it creates a mapping of frames and labels
- create_database.py: Extracts the features of the frames, where a batch of 5 frames was evaluated at the same time for one sample in the resulting database (Muppets-02-01-01-video.csv, Muppets-02-04-04-video.csv, Muppets-03-04-03-video.csv)
- create_sound.py: Extracts the sound from the videos, then creates frames from it and detects the features, which are saved in a dedicated database (Muppets-02-01-01-sound.csv, Muppets-02-04-04-sound.csv, Muppets-03-04-03-sound.csv). Afterwards it merges the two Databases into Muppets-02-01-01-final.csv, Muppets-02-04-04-final.csv and Muppets-03-04-03-final.csv
- build.py: Runs all the methods from above to set the database up. WARNING: THIS MAY TAKE OVER 80 HOURS
- classify_data.py: Trains three different classifiers on the final database and creates a statistic for the purpose of analyzing the results

## Techniques Used
### Video Features
We created a batch of 5 frames to be analyzed as one sample, like a sliding window. For each Frame we calculated:
- Number of pixels that are colored like a pig
- Texture of the image with selected Haralick features
- Optical flow
For the pig colors we first analyzed pictures of Miss Piggy and selected the 20 most common color values in different lighting environments. This was done in the analyse_mp() method in create_database.py together with an analysis of the Haralick features. Since we wanted to keep the information in the distribution of each value, each feature or set of features occurrs five times in a row. As the usual Haralick features count 13 different features, we selected only 7 that were analyzed experimentally. The optical flow was calculated by first applying edge detection to find points that were easy to track. These points were updated in each iteration. From each picture to the next we calculated the movement vectors and added the absolute length of them. This makes four values to determine how fast a picture is.

### Audio Features
As written earlier, the sound track was split into multiple 'frames' of 200 miliseconds. Each frame was then first transformed with the fast fourier transformation to get the power levels. We used 256 points for this purpose. Afterwards we created the MEL frequency bank with 40 bins and applied it to the power levels. The spectrum was then transformed back with MEL-frequency Cepstral Coefficients with a target of 20 values.

### Classification
We wanted to try out different classifiers for this project and ended up with:
- Linear SVM
- kNN
- Decision Tree
All from the SciKit-Learn library. Training and testing was done with 10-folded cross-validation. The clear winner was the decision tree with an accuracy of 92%, a precision of 79% and recall of 79%. A big problem for the classification was the very small number of occurrances of pigs inside the movies, which led to some experiments with the SVM that had an accuracy of 82%, which sounded great but a recall of only 1%. In this case the SVM simply assigned 0 to almost all of the samples, which would not be visible as the accuracy was still very high. Similarly kNN produced solid accuracies with very low values of recall. In this algorithm we used a k of 5, since eventhough the sample size was very high, using too many neighbors would mean that it is more likely to get disturbed by noise from the negative samples.



 
