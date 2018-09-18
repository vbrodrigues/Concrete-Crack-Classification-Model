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

IMG_SIZE = 50

def prepare_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model(".../Concrete_Crack_Classification_model.model") #Replace the dots with the directory

#Insert your image file inside the double quotes
prediction = model.predict([prepare_image("")])

if prediction[0][0] == 0:
    print("\nEste concreto POSSUI uma fissura.")
elif prediction[0][0] == 1:
    print("\nEste concreto NÃO POSSUI uma fissura.")
else:
    print("\nOcorreu algum erro...")
