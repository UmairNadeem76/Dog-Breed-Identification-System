import numpy as np
import cv2

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)

trainMyImageFolder = "C:/Users/shahz/OneDrive/Desktop/AI Project/train"
# load the csv file
import pandas as pd

df = pd.read_csv('C:/Users/shahz/OneDrive/Desktop/AI Project/labels.csv')
print("head of labels :")
print("================")

print(df.head())
print(df.describe())

print("Group by labels : ")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))

img_path = "C:/Users/shahz/OneDrive/Desktop/AI Project/train/2a3b3a4fecb3171df19bed491865c733.jpg"
img = cv2.imread(img_path)
#cv2.imshow("img", img)
#cv2.waitKey(0)

allImages = []
allLabels = []
import os

from keras.preprocessing.image import img_to_array, load_img

for image_name, breed in df[['id', 'breed']].dropna().values:
    img_path = os.path.join(trainMyImageFolder, str(image_name) + '.jpg')
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    allImages.append(img_array)
    allLabels.append(breed)


print("saving the data :")
np.save("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsImages.npy",allImages)
np.save("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsLabels.npy",allLabels)

print("finish save the data ....")