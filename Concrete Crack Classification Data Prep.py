print("Importing libraries...")

import cv2
import numpy as np
import os
import random
import h5py

data_directory = "" #insert the directory you'll be working with
img_size = 128
categories = ["Positive", "Negative"]
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)
        
        # read and resize the images and append to training_data a list with the image itself and its class number
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])

print("Creating training data...")
create_training_data()
print("Training data successfully created!!")

print("Shuffling training data...")
random.shuffle(training_data)
print("Training data successfully shuffled!!")

X_data = []
y = []

# create X with the features (the images) and y with the targets (labels)
for features, label in training_data:
    X_data.append(features)
    y.append(label)

print("X and y data successfully created!!")

# reshape the image to be on the correct format for tensorflow (nº images, width, height, channels)
print("Reshaping X data...")
X = np.array(X_data).reshape(len(X_data), img_size, img_size, 1)
print("X data successfully reshaped!!")

print("Saving the data...")
hf = h5py.File(".../concrete_crack_image_data.h5", "w) #Replace the three dots with the directory you want to save your dataset in
hf.create_dataset("X_concrete", data = X, compression = "gzip")
hf.create_dataset("y_concrete", data = y, compression = "gzip")
hf.close()
print("Data successfully saved!!")
