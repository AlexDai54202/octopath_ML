from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import sklearn
import numpy as np
import sys
import os
import json
import re
from PIL import Image

import tensorflow as tf
import keras

import keras.utils as image


df = pd.read_csv('dataset.csv')


def arrange_data(dfin):
    
    image_data = []
    img_paths = np.asarray(dfin['asset_path'])
    
    for i in range(len(img_paths)):
        img = image.load_img(img_paths[i],target_size=(128, 128, 3))
        img = image.img_to_array(img)
        img = img/255
        image_data.append(img)
        
        
    X = np.array(image_data)
    Y = np.array(dfin[['Fire','Ice','Lightning','Wind','Light','Dark','Sword','Spear','Dagger','Axe','Bow','Staff']])
    
    print("Shape of images:", X.shape)
    print("Shape of labels:", Y.shape)
    
    return X, Y

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.1)



x_train, y_train = arrange_data(dfin = train)


x_test, y_test = arrange_data(dfin = test)

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# freeze last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


model = models.Sequential()

model.add(vgg_conv)

num_classes = 12

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS=50
BS = 64

# data augmentation to reduce overfit
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")

history = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BS, epochs=EPOCHS)

model.save('Model_4d.h5')


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))


plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.savefig('accuracy.png')

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


plt.savefig('cross_entropy_loss.png')