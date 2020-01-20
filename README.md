


# Similarity Modeling  1 & 2
Implementation of Similarity Modeling Projects  by Nikolaus Funk and Bassel Mahmoud  
  
## 1. Problem  
- SM1: Detection of the Kermit in the given video files using both audio and visual features and Deep neural networks.
- SM2 - Detection of the pigs in the given video files using both audio and visual features.
  
## 2. Libraries used  
- Numpy  
- Mahotas  
- Sci-kit Learn  
- OpenCV  
- Librosa  
- Moviepy.editor  
- Pydub 
- The rest of the requirements are defined in requirnment.txt
  
## 3. Usage
### 3.1. SM1: Deep Learning approach
The sources and scripts for visual-based DL and auditory based DL can be found in sm1-deep-video and sm1-deep-audio respectively.
To be able to run the following script the videos and label data should be in the data folder. We recommend  to create a separate python environment and install the requirement using
`pip install -r requirements.txt`
#### 3.1.1. Visual DL

- helpers/generateFrames.py: will extract 5 frames per seconds
- model1.py: the used model with no regulation, which caused the model to overfit.
- model2.py: the better model that included Dropout layers and regulation, we also added the finetune method
- train.py: will load the data and perform the training.
- evaluate.py: will evaluate a saved model.
- test.py: will play a video and classify the frames in real-time. (Videos should be in the data folder and renamed to, video1.avi, video2.avi, and video3.avi)
To run the training/evaluating, run one the following command after importing the labeled data to the data folder
`python train.py` or `python evaluate.py`

#### 3.1.2. auditory DL
- helpers/audioloader.py: extend the Iterator class from Keras, which will load the audio tracks and precess them to get a spectrogram using Librosa.
- helpers/create_sounds.py: will extract the audio from the video and split it into 1 second overlapped tracks. The labeling was done by hand with the use of the labeling function in [Audacity](https://www.audacityteam.org/)
- model.py: contains the model definition, we used 3 Conv2d layers with 2 fully connected layers.
- train.py: will start the training of the model.
- evaluate.py: will evaluate a saved model.
To run the training/evaluating, run one the following command after importing the labeled data to the data folder
`python train.py` or `python evaluate.py`

The trained models for both approaches can be downloaded from:
[Dropbox shared folder to the models](https://www.dropbox.com/sh/8npr9lwuk7wb3ey/AADF3qOmkRHtnO2Xx0oh7h9ia?dl=0)
### 3.2. SM2: Classical approach
The implementation is located in sm2-classical folder.
This Project is split into  
- create_frames.py: Extracts the frames from the video data  
- create_labels.py: Based on the label and timestamp lists (m_02_01_01_pig_labels, m_02_04_04_pig_labels, m_03_04_03_pig_labels) it creates a mapping of frames and labels  
- create_database.py: Extracts the features of the frames, where a batch of 5 frames was evaluated at the same time for one sample in the resulting database (Muppets-02-01-01-video.csv, Muppets-02-04-04-video.csv, Muppets-03-04-03-video.csv)  
- create_sound.py: Extracts the sound from the videos, then creates frames from it and detects the features, which are saved in a dedicated database (Muppets-02-01-01-sound.csv, Muppets-02-04-04-sound.csv, Muppets-03-04-03-sound.csv). Afterward, it merges the two Databases into Muppets-02-01-01-final.csv, Muppets-02-04-04-final.csv, and Muppets-03-04-03-final.csv  
- build.py: Runs all the methods from above to set the database up. WARNING: THIS MAY TAKE OVER 80 HOURS  
- classify_data.py: Trains three different classifiers on the final database and creates a statistic to analyze the results  
  
To classify running classify_data.py should be sufficient, as the other files are only dedicated to the creation of the database that is already handed in together with the implementation.  
  
## 4. Techniques and Experiments
### 4.1. SM1
#### 4.1.1. Video DL
In the visual approach, we used transfer learning since the amount of data is relatively small. We imported the pre-trained Conv layers from VGG16 network to use as feature extraction, after those layers, three fully connected layers were added to perform the classification.
Data preprocessing: 5 frames/second were extracted from the videos and labeled, after that, we used the same preprocessing that is done in VGG16 since we used the same first 20 layers.
The training was done in two steps:
- The first one trained the fully connected layers with a high learning rate.
- The second step, trained the Conv layers with a lower learning rate (baseLearningRate/10), the purpose of this step to fine-tune the Conv layers.
Transfer learning helped to reduce the training time by a huge factor, we only needed 15 mins.
##### Results
model-8-fineTuned on all the data:

| Classes      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 1            | 0.96      | 0.99   | 0.98     | 6265    |
| 0            | 1.00      | 0.99   | 0.99     | 16919   |
| accuracy     |           |        | 0.99     | 23184   |
| macro avg    | 0.98      | 0.99   | 0.98     | 23184   |
| weighted avg | 0.99      | 0.99   | 0.99     | 23184   |

model-8-fineTuned on test data:

| Classes      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 1            | 0.86      | 0.95   | 0.90     | 1253    |
| 0            | 0.98      | 0.94   | 0.96     | 3383    |
| accuracy     |           |        | 0.94     | 4636    |
| macro avg    | 0.92      | 0.95   | 0.93     | 4636    |
| weighted avg | 0.95      | 0.94   | 0.95     | 4636    |

We can see that the results using transfer learning are very good, except the case of precision when Kermit is visible in the frame, it dropped to 0.86. We think the reason for that is the low quality of the video and the motion blur that is highly visible when transitioning from frame to frame.

#### 4.1.2. Audio DL
In this part, we were inspired by [SincNet](https://arxiv.org/abs/1808.00158) paper to use Convolution neural networks to classify raw audio files. we applied Fourier transform on the tracks to get a spectrum (The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows) then we get used the magPhase function from librosa to separate the spectrogram into its magnitude and phase components, this results in a 2D array which can be seen as an image.
We have tested multiple configuration and hyperparameters for the neural networks, but we did not accomplish good results comparing to the video DL, we think that issue is related to the labeling of the tracks and the background noise when Kermit is speaking.
##### Results

| Classes      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 1            | 0.84      | 0.83   | 0.83     | 523     |
| 0            | 0.97      | 0.97   | 0.97     | 3352    |
| accuracy     |           |        | 0.96     | 3875    |
| macro avg    | 0.96      | 0.96   | 0.96     | 3875    |
| weighted avg | 0.91      | 0.90   | 0.90     | 3875    |

### 4.2. SM2
#### 4.2.1. Video Features  
We created a batch of 5 frames to be analyzed as one sample, like a sliding window. For each Frame we calculated:  
- Number of pixels that are colored like a pig  
- The texture of the image with selected Haralick features  
- Optical flow  
  
For the pig colors, we first analyzed pictures of Miss Piggy and selected the 20 most common color values in different lighting environments. This was done in the analyse_mp() method in create_database.py together with an analysis of the Haralick features. Since we wanted to keep the information in the distribution of each value, each feature or set of features occurs five times in a row. As the usual Haralick features count 13 different features, we selected only 7 that were analyzed experimentally. The optical flow was calculated by first applying edge detection to find points that were easy to track. These points were updated in each iteration. From each picture to the next we calculated the movement vectors and added the absolute length of them. This makes four values to determine how fast a picture is.  
  
#### 4.2.2. Audio Features  
As written earlier, the soundtrack was split into multiple 'frames' of 200 milliseconds. Each frame was then first transformed with the fast Fourier transformation to get the power levels. We used 256 points for this purpose. Afterward, we created the MEL frequency bank with 40 bins and applied it to the power levels. The spectrum was then transformed back with MEL-frequency Cepstral Coefficients with a target of 20 values.  
  
#### 4.2.3. Classification  
We wanted to try out different classifiers for this project and ended up with:  
- Linear SVM  
- KNN  
- Decision Tree  
  
All from the SciKit-Learn library. Training and testing were done with 10-folded cross-validation. The clear winner was the decision tree with an accuracy of 92%, a precision of 77% and a recall of 77%. A big problem for the classification was the very small number of occurrences of pigs inside the movies, which led to some experiments with the SVM that had an accuracy of 82%, which sounded great but a recall of only 1%. In this case, the SVM simply assigned 0 to almost all of the samples, which would not be visible as the accuracy was still very high. Similarly, KNN produced solid accuracies with very low values of recall. In this algorithm, we used a k of 5 since even though the sample size was very high, using too many neighbors would mean that it is more likely to get disturbed by noise from the negative samples. A detailed statistic can be found with 150 entries for accuracy, precision and recall for each model in result_SVM.csv, result_kNN.csv, and result_DecTree.csv.  
  
These experiments were conducted on 30% of the overall database consisting of all three datasets. Averaging the experiments shows the following results:  
  
Linear SVM:  
- Accuracy: 0.728  
- Precision: 0.242  
- Recall: 0.223  
  
KNN:  
- Accuracy: 0.83  
- Precision: 0.561  
- Recall: 0.306  
  
Decision Tree:  
- Accuracy: 0.916  
- Precision: 0.769  
- Recall: 0.77  
  
  

## 5. Conclusion
The obvious result is DL produced better results than classical methods with less time and easier implementation.
### 5.1. SM1
The interesting part of the video DL that, we have used transfer learning, which proved to be efficient when having a small dataset and reducing the training time, which helped to test multiple configurations quickly. The Audio DL helped us understand Fourier transform, additionally how to use convolutional neural networks with audio data.
### 5.2. SM2 
The very good result with the decision tree model was very exciting for us since it was hard for us to see whether our tests and analyses were correct and would lead to a sufficiently good model. The most tedious part of our work was the experiments with different textures and colors, as well as the error handling of the optical flow method. While it was the most interesting part of this project, finding out how speech recognition works were by far the hardest to get a grasp on.


## 6. Resources and References
- [SincNet paper](https://arxiv.org/abs/1808.00158) Speaker Recognition from Raw Waveform with SincNet
- [SincNet](https://github.com/mravanelli/SincNet) Implementation
- [A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) 
- [VGG16](https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/)  Understanding various architectures of Convolutional Networks
