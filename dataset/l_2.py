import tensorflow as tf
import cv2
from pathlib import Path

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.contrib.data import Dataset, Iterator
image = cv2.imread("/home/mihael/Documents/9. semestar/VIROKR/Projekt/dataset/i002qa-mn.jpg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
file = Path("haarcascade_frontalface_default.xml")
print(file.exists())
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

bounding_boxes = face_cascade.detectMultiScale(grayscale_image, 1.25, 6)

model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(96, 96, 1), dim_ordering="tf"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(36, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(48, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D())

model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(30))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath="face_model.h5", verbose=1, save_best_only=True)

epochs = 30

hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)