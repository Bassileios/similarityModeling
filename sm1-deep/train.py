import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.metrics import Accuracy, Recall, Precision
import tensorflow.keras.backend as K
from helpers.generateFrames import generate_all_frames
from model import create_model
from model2 import create_model2, finetune_model
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # for mathematical operations
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize  # for resizing images
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.metrics import classification_report



def MSG(txt):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), str(txt))


MODEL_NAME = 'model-11'
USE_MODEL2 = True
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
LEARNING_RATE = 0.0001


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    data_dir = pathlib.Path('../data/kermit/')
    if not os.path.exists('../data/video1'):
        MSG("No frames found, extracting frames from videos")
        generate_all_frames()

    image_count = len(list(data_dir.glob('*/*.jpg')))
    MSG('image count = ' + str(image_count))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    MSG("classes: " + str(CLASS_NAMES))

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                      horizontal_flip=True,
                                                                      validation_split=0.2,
                                                                      preprocessing_function=preprocess_input)

    # image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
    #                                                                   shear_range=0.2,
    #                                                                   zoom_range=0.2,
    #                                                                   horizontal_flip=True,
    #                                                                   validation_split=0.2,
    #                                                                   preprocessing_function=preprocess_input)

    STEPS_PER_EPOCH = np.ceil(0.8 * image_count / BATCH_SIZE)
    VALIDATION_STEPS_PER_EPOCH = np.ceil(0.2 * image_count / BATCH_SIZE)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES),
                                                         subset='training')

    validation_generator = image_generator.flow_from_directory(directory=str(data_dir),
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               classes=list(CLASS_NAMES),
                                                               subset='validation')

    image_batch, label_batch = next(train_data_gen)
    show_batch(image_batch, label_batch, CLASS_NAMES)

    # x_train, x_valid, y_train, y_valid = prepare_data()

    MSG("Creating model")
    if USE_MODEL2:
        model = create_model2()
    else:
        model = create_model()

    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    # model.compile(optimizer='Adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # metrics = ['accuracy', Precision(), Recall()]
    metrics = ['accuracy', precision, recall, f1_score]

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=metrics)
    model.summary()
    MSG(len(model.trainable_variables))

    log_dir = "logs/fit/" + MODEL_NAME + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=10,
                        validation_data=validation_generator,
                        validation_steps=VALIDATION_STEPS_PER_EPOCH,
                        callbacks=[tensorboard_callback])
    model.save('model/' + MODEL_NAME)

    finetune_model(model)

    # model.compile(optimizer='Adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE / 10),
                  metrics=metrics)
    model.summary()
    log_dir = "logs/fit/" + MODEL_NAME + '-tuned' + datetime.now().strftime("%Y%m%d-%H%M%S")

    MSG(len(model.trainable_variables))

    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=10,
                        validation_data=validation_generator,
                        validation_steps=VALIDATION_STEPS_PER_EPOCH,
                        callbacks=[tensorboard_callback])

    model.save('model/' + MODEL_NAME + '-tuned')

    predictions = model.predict_generator(validation_generator, steps=VALIDATION_STEPS_PER_EPOCH)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())

    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)



def prepare_data():
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
    x_train, x_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3,
                                                          random_state=42)  # preparing the validation set
    return x_train, x_valid, y_train, y_valid


def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n] == 1][0].title())
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
