from datetime import datetime
from helpers.audioloader import SpeechDirectoryIterator
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from model import get_model
import tensorflow as tf
from sklearn.metrics import classification_report

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_path = '../data/kermit-sound/'
test_path = '../data/kermit-sound/'
# The classes correspond to directory names under ../train/audio
CLASS_NAMES = ['0', '1']

MODEL_NAME = 'model-11'

window_size = .02
window_stride = .01
window_type = 'hamming'
normalize = True
max_len = 101
batch_size = 64
log_dir = "logs/fit/" + MODEL_NAME + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def main():
    # loading the trained model
    my_model = tf.keras.models.load_model('model/model-8-tuned')
    my_model.summary()

    test_iterator = SpeechDirectoryIterator(directory=train_path,
                                            batch_size=batch_size,
                                            window_size=window_size,
                                            window_stride=window_stride,
                                            window_type=window_type,
                                            normalize=normalize,
                                            max_len=max_len,
                                            classes=CLASS_NAMES,
                                            shuffle=True,
                                            seed=123)

    predictions = my_model.predict_generator(test_iterator,
                                             steps=np.ceil(test_iterator.n / batch_size))
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_iterator.classes
    class_labels = list(test_iterator.class_indices.keys())

    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)


if __name__ == '__main__':
    main()
