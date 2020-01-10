from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helpers.generateFrames import generate_all_frames
from model import create_model
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np    # for mathematical operations
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize   # for resizing images

def MSG(txt):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))


def main():
    if not os.path.exists('../data/video1'):
        MSG("No frames found, extracting frames from videos")
        generate_all_frames()

    data = pd.read_csv('../mapping.csv')
    print(data.head())

    X = []  # creating an empty array
    for img_name in data.Image_ID:
        img = plt.imread('../data/' + 'video1/' + img_name)
        X.append(img)  # storing each image in array X
    X = np.array(X)  # converting list to array
    y = data.Class
    dummy_y = to_categorical(y)  # one hot encoding Classes

    image = []
    for i in range(0, X.shape[0]):
        a = resize(X[i], preserve_range=True, output_shape=(224, 224)).astype(int)  # reshaping to 224*224*3
        image.append(a)
    X = np.array(image)

    X = preprocess_input(X, mode='tf')

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set

    MSG("Creating model")
    model = create_model()

    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid))

if __name__ == '__main__':
    main()
