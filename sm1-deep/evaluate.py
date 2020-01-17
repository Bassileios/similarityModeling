import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import pathlib
import numpy as np
from sklearn.metrics import classification_report

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_dir = pathlib.Path('../data/kermit/')
    # loading the trained model
    my_model = tf.keras.models.load_model('model/model-8')
    my_model.summary()

    image_count = len(list(data_dir.glob('*/*.jpg')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)


    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                      horizontal_flip=True,
                                                                      preprocessing_function=preprocess_input)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES))
    # my_model.evaluate(train_data_gen)

    predictions = my_model.predict_generator(train_data_gen, steps=STEPS_PER_EPOCH)
    predicted_classes = np.argmax(predictions, axis=1)


    true_classes = train_data_gen.classes
    class_labels = list(train_data_gen.class_indices.keys())
    print(predicted_classes)
    neg = np.logical_not(predicted_classes).astype(int)
    print(neg)
    print(true_classes)
    print(class_labels)

    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)
    report = classification_report(true_classes, neg, target_names=class_labels)
    print(report)


if __name__ == '__main__':
    main()
