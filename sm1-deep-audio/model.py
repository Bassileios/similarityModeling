from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2

def get_model(inputshape, outputshape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=inputshape, kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(outputshape, activation='softmax'))  # Last layer with one output per class
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    model.summary()
    return model

