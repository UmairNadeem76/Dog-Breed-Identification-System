import tensorflow as tf
import os
import numpy as np
import cv2

modelFile = "C:/Users/shahz/OneDrive/Desktop/AI Project/dogs.h5"
model = tf.keras.models.load_model(modelFile)

#print(model.summary() )

inputShape = (331,331)

allLabels = np.load("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsLabels.npy")
categories = np.unique(allLabels)

def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.
    return imgResult

#testImagePath = "C:/Users/shahz/OneDrive/Desktop/dog-breed-identification/train/02b1c50fb7315423a664f3ce68c94e30.jpg"
testImagePath = "C:/Users/shahz/OneDrive/Desktop/TestImages/kelpie.jpg"

#load image
img = cv2.imread(testImagePath)
imageForModel = prepareImage(img)

# predicition

resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis = 1)

print(answers)

text = categories[answers[0]]

print(text)

desired_size = (500, 500)

# Resize the image to the desired size
img = cv2.resize(img, desired_size)

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text , (0,20), font , 0.5, (209,19,77), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()