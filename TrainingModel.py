import numpy as np

IMAGE_SIZE = (331,331)
IMAGE_FULL_SIZE = (331,331,3)
batchSize = 8

allImages = np.load("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsImages.npy")
allLabels = np.load("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsLabels.npy")

print(allImages.shape)
print(allLabels.shape)

# convert the labels text to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
integerLabels = le.fit_transform(allLabels)
print(integerLabels)

# unique integer labels
numOfCategories = len(np.unique(integerLabels))  
print(numOfCategories)

# convert the integer labels to categorical -> prepare for the train
from keras.utils import to_categorical

allLabelsForModel = to_categorical(integerLabels, num_classes = numOfCategories)
print(allLabelsForModel)

# normalize the images from 0-255 to 0-1
allImagesForModel = allImages / 255.0


# create train and test data
from sklearn.model_selection import train_test_split

print("Before split train and test :")

X_train , X_test , y_train , y_test = train_test_split(allImagesForModel, allLabelsForModel, test_size=0.3, random_state=42)

print("X_train , X_test , y_train , y_test -----> shapes :")

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# free some memory
del allImages
del allLabels
del integerLabels
del allImagesForModel

# build the model

from keras.layers import Dense , Flatten
from keras.models import Model
from keras.applications.nasnet import NASNetLarge

myModel = NASNetLarge(input_shape=IMAGE_FULL_SIZE , weights='imagenet', include_top=False)

for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

plusFlattenLayer = Flatten()(myModel.output)

predicition = Dense(numOfCategories, activation='softmax')(plusFlattenLayer)

model = Model(inputs=myModel.input, outputs=predicition)


from keras.optimizers import Adam

lr = 1e-4 # 0.0001
opt = Adam(lr)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = opt,
    metrics=['accuracy'] )

stepsPerEpoch = np.ceil(len(X_train) / batchSize)
validationSteps = np.ceil(len(X_test) / batchSize)


from keras.callbacks import ModelCheckpoint , ReduceLROnPlateau , EarlyStopping

best_model_file = "C:/Users/shahz/OneDrive/Desktop/AI Project/dogs.h5"

callbacks = [
        ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1 , verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_accuracy', patience=7, verbose=1) ]

r = model.fit (
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs = 10,
    batch_size = batchSize,
    steps_per_epoch=stepsPerEpoch,
    validation_steps=validationSteps,
    callbacks=[callbacks]
)