#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:13:29 2017

@author: wrutka
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

# Initialising the CNN

def build_classifier(nb_ann_layers=1, nb_cnn_layers=1, ann_activation='relu', cnn_activation='relu'):
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    for i in range(nb_cnn_layers):
        classifier.add(Conv2D(32, (3, 3), activation = cnn_activation))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
    #
    ## Adding a second convolutional layer
    #classifier.add(Conv2D(98, (3, 3), activation = 'relu'))
    #classifier.add(MaxPooling2D(pool_size = (4, 4)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    for i in range(nb_ann_layers):
        classifier.add(Dense(units = 256, activation = ann_activation))
        classifier.add(Dropout(0.6))

    classifier.add(Dense(units = 2, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('CNN_datatest_cropped/train',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 color_mode = 'rgb',
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('CNN_datatest_cropped/test',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            color_mode = 'rgb',
                                            class_mode = 'categorical')

params = {
        'nb_ann_layers': [4],
        'nb_cnn_layers': [4],
        }
params_attr = params.copy()

#import itertools as it
#combinations = it.product(*(v for _, v in params.items()))

combinations = []
for k, v in params.items():
    params_attr.pop(k)
    for v_item in v:
        for second_k, second_v in params_attr.items():
            if second_k == k:
                continue
            for second_v_item in second_v:
                combinations.append({k: v_item, second_k: second_v_item})


epochs = 10
results = []
for param in combinations:
    classifier = build_classifier(**param)
    res = classifier.fit_generator(training_set,
                         steps_per_epoch = 35,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 15)
    results.append((classifier, param, 'acc: %f, loss: %f, val_acc: %f, val_loss: %f'
                    %(res.history['acc'][epochs-1], res.history['loss'][epochs-1],
                      res.history['val_acc'][epochs-1], res.history['val_loss'][epochs-1])))

for res in results:
    ann_layers = res[1]['nb_ann_layers']
    cnn_layers = res[1]['nb_cnn_layers']
#    res[0].save('FR_model_ann_layers_%d_cnn_layers_%d-%s.hdf5' % (ann_layers, cnn_layers, res[2]))
    res[0].save('categorical_crossentropy.hdf5')

