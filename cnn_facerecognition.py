#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:13:29 2017

@author: wrutka
"""

import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from glob import glob
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.pooling import AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras import optimizers 
from keras.callbacks import ModelCheckpoint, History, Callback
from keras.layers.advanced_activations import PReLU
from keras import initializations
from keras.models import load_model
from keras.optimizers import *


# Settings {{{
labels_map = {
        'kinga': 0,
        'wiktor': 1
        }
#}}}


dataset_paths = glob('./CNN_datatest_cropped/*')

image_data = []
image_labels = []

for i, fl in enumerate(dataset_paths):
    img_read = io.imread(fl, cv2.IMREAD_GRAYSCALE)
    img_read = cv2.resize(img_read, (200, 200), cv2.IMREAD_GRAYSCALE)
    
    #
    if len(img_read.shape) == 3:
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    
    image_data.append(img_read)
    
    ##translate kinga  => 0
    ##          wiktor => 1
    image_label, label_read = os.path.split(fl)[1].split(".")[0].split('_')[:2]
    image_labels.append(int(image_label))


#plt.imshow(image_data[4], cmap='gray')

## Train test split
X = np.array(image_data)
y = np.array(image_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    train_size=0.8, random_state = 20)

nb_classes = len(np.unique(y))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.
X_test /= 255.

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

def build_model(size):
    model = Sequential()
    init = 'he_normal'
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', 
                            init=init, input_shape=(1, size, size), subsample=(1, 1) ))
#    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', init=init))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool'))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1', init=init))
#    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', init=init))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool'))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', init=init))
#    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', init=init))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', init=init))
#    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', init=init))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block4_pool'))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu', init='he_normal', name='fc1'))
    model.add(Dropout(p=0.5))
    model.add(Dense(500, activation='relu', init='he_normal', name='fc2'))
    model.add(Dropout(p=0.5))
#    model.add(AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='th'))
    #model.add(Dropout(p=0.5))
    model.add(Dense(15, activation='softmax', name='predictions'))
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optim = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model