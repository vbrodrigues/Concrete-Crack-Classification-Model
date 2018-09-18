import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import random
import pickle

DATADIR = "" #insert the directory you'll be working with
IMG_SIZE = 50
CATEGORIES = ["Positive", "Negative"]
training_data = []

print("Loading the data...")
pickle_in = open(".../X_concrete.pickle", "rb") #Replace the dots with the directory
X = pickle.load(pickle_in)
pickle_in = open(".../y_concrete.pickle", "rb") #Replace the dots with the directory
y = pickle.load(pickle_in)
print("Data successfully loaded!")

X = X / 255.0

model = Sequential()

model.add(Conv2D(128, (3, 3), activation = "relu", input_shape = (IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(258, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
model.fit(X, y, batch_size = 128, epochs = 3, validation_split = .2)
print("Model successfully fitted!!")

print("Saving the model...")
model.save(".../Concrete_Crack_Classification_model.model") #Replace the dots with the directory
print("Model successfully saved!!")
