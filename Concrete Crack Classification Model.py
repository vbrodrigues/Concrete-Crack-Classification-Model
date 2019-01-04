print("Importing libraries...")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import h5py

img_size = 128

print("Loading the data...")
hf = h5py.File('.../concrete_crack_image_data.h5', 'r') #Replace the three dots with the directory you saved the dataset in
X = np.array(hf.get('X_concrete'))
y = np.array(hf.get("y_concrete"))
hf.close()
print("Data successfully loaded!")

print("Scaling the data...!")
X = X / 255
print("Data successfully scaled!")

model = Sequential()

model.add(Conv2D(16, (3, 3), activation = "relu", input_shape = (img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(258, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
model.fit(X, y, batch_size = 64, epochs = 5, validation_split = .2)
print("Model successfully fitted!!")

print("Saving the model...")
model.save(".../Concrete_Crack_Classification_model.model") #Replace the dots with the directory you want to save the model in
print("Model successfully saved!!")
