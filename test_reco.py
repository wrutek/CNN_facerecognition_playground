#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:24:46 2017

@author: wrutka
"""
from keras.preprocessing import image
from keras.models import load_model
from glob import glob

files = glob('test_img/*')
classifier = load_model('FR_model_ann_layers_4_cnn_layers_4-acc_1.000000_loss_0.001747_val_acc_1.000000_val_loss_0.008827.hdf5')

import numpy as np

results = []
for file in files:
    img = image.load_img(file, target_size=(200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    res = classifier.predict(img)
    results.append(file + " - " + str(res))
    print(file)
    print(res)
    