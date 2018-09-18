print("Importing libraries...")

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

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

print("Creating training data...")
create_training_data()
print("Training data successfully created!!")

print("Shuffling training data...")
random.shuffle(training_data)
print("Training data successfully shuffled!!")

X_data = []
y = []

for features, label in training_data:
    X_data.append(features)
    y.append(label)

print("X and y data successfully created!!")

print("Reshaping X data...")
X = np.array(X_data).reshape(len(X_data), IMG_SIZE, IMG_SIZE, 1)
print("X data successfully reshaped!!")

print("Saving the data...")
pickle_out = open(".../X_concrete.pickle", "wb") #Replace the dots with the directory
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(".../y_concrete.pickle", "wb") #Replace the dots with the directory
pickle.dump(y, pickle_out)
pickle_out.close()
print("Data successfully saved!!")
