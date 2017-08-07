#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:22:06 2017

@author: wrutka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:13:29 2017

@author: wrutka
"""

import cv2
import os

from skimage import io
from glob import glob
import matplotlib.pyplot as plt

# Settings {{{
labels_map = {
        'kinga': 0,
        'wiktor': 1
        }
#}}}


dataset_paths = glob('./CNN_datatest/*')

image_data = []
image_labels = []

for fl in dataset_paths:
    img_read = io.imread(fl)
    img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    image_data.append(img_read)    
    
    #translate kinga  => 0
    #          wiktor => 1
    label_read = os.path.split(fl)[1].split(".")[0].split('_')[0]
    image_labels.append(labels_map[label_read])


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


xs_clahe = []
cropped = []
cropped_labels = []
cropped_names = []
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(31, 31))
for i, im in enumerate(image_data):
    x_clahe = clahe.apply(im)
    xs_clahe.append(x_clahe)
    class_ = face_cascade.detectMultiScale(im)
    
    if len(class_):
        x, y, w, h = class_[0]

        cropped.append(x_clahe[y:y+h, x:x+w])
        cropped_labels.append(image_labels[i])
        cropped_names.append(dataset_paths[i])

for i, c in enumerate(cropped):
    cv2.imwrite('./CNN_datatest_cropped/'+str(cropped_labels[i])+'_'+os.path.split(cropped_names[i])[1], c)
#plt.imshow(x_clahe, cmap='gray')
