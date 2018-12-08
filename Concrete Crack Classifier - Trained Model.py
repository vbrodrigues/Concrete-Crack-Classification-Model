import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 128
img_to_prepare = "" #Insert the path to the image file you want to predict

def prepare_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Loading trained model...")
model = tf.keras.models.load_model(".../Concrete_Crack_Classification_model.model") #Replace the dots with the directory you saved the model in
print("Trained model loaded!")

print("Model predicting...")
#Insert your image file inside the double quotes
prediction = model.predict([prepare_image(img_to_prepare)])

if prediction[0][0] <= .5:
    pred_text = "Networks prediction:\nThis surface DOES have a crack on it."
elif prediction[0][0] > .5:
    pred_text = "Networks prediction:\nThis surface DOES NOT have a crack on it."
else:
    print("\nSomething went wrong...")
    
plt.imshow(cv2.resize(cv2.imread(img_to_predict), (IMG_SIZE, IMG_SIZE)))
plt.title("What the Neural Network is receiving as input:")
plt.text(2, 5, pred_text, fontweight = "bold")
plt.show()
