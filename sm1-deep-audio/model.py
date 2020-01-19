from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def get_model(inputshape, outputshape):
    model = Sequential()
    model.add(Conv2D(12, (5, 5), activation='relu', input_shape=inputshape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(25, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(180, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(outputshape, activation='softmax'))  # Last layer with one output per class
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    model.summary()
    return model

