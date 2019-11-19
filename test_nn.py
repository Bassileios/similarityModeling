import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_img, train_lab), (test_img, test_lab) = data.load_data()

#train_img = train_img/255
#test_img = test_img/255

model = keras.Sequential([
	keras.layers.Input(28*28),
	#keras.layers.Conv2D(),
	keras.layers.Dense(128,activation='relu'),
	keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_img,train_lab,test_img,test_lab)

test_loss, test_acc = model.evaluate(test_img, test_lab)

print('Test Acc',test_acc)
