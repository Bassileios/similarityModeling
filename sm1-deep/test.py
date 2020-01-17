import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.transform import resize
import numpy as np


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # loading the trained model
    my_model = tf.keras.models.load_model('model/model-8-tuned')
    my_model.summary()

    vid_name = '../data/video3.avi'
    cap = cv2.VideoCapture(vid_name)
    framerate = cap.get(5)  # frame rate
    cv2.namedWindow('video1', cv2.WINDOW_AUTOSIZE)
    label = 'No'
    while True:
        ret_val, frame = cap.read()
        images = []
        a = prepare_frame(frame)
        images.append(a)
        test_image = np.array(images)
        frameID = cap.get(1)


        if frameID % framerate//5 == 0:
            pred = my_model.predict(test_image)
            pred_class = np.argmax(pred, axis=1)

            label = 'Yes' if pred_class == 0 else 'No'

        # draw the label into the frame
        __draw_label(frame, label, (20, 20), (255, 0, 0))

        cv2.imshow('video1', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def prepare_frame(frame):
    a = resize(frame, preserve_range=True, output_shape=(224, 224)).astype(int)
    a = preprocess_input(a, mode='tf')
    return a


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


if __name__ == '__main__':
    main()
