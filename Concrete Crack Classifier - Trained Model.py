import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 128

def prepare_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Loading trained model...")
model = tf.keras.models.load_model(".../Concrete_Crack_Classification_model.model") #Replace the dots with the directory
print("Trained model loaded!")
print("Model evaluation after training: loss: 0.0528 - acc: 0.9828 - val_loss: 0.0573 - val_acc: 0.9860")

print("Model predicting...")
#Insert your image file inside the double quotes
prediction = model.predict([prepare_image("")])

if prediction[0][0] <= .5:
    pred_text = "Previsão da Rede:\nEsta superfície de concreto POSSUI uma trinca."
    # print("\nEste concreto POSSUI uma fissura.")
elif prediction[0][0] > .5:
    pred_text = "Previsão da Rede:\nEsta superfície de concreto NÃO POSSUI uma trinca."
    # print("\nEste concreto NÃO POSSUI uma fissura.")
else:
    print("\nOcorreu algum erro...")
    
plt.imshow(cv2.resize(cv2.imread(img_to_predict), (IMG_SIZE, IMG_SIZE)))
plt.title("O que a Rede Neural está recebendo como entrada:")
plt.text(2, 5, pred_text, fontweight = "bold")
plt.show()
