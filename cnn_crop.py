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


dataset_paths = glob('./new_wiktor/*')

image_data = []
image_data_gray = []
image_labels = []

for fl in dataset_paths:
    img_read = io.imread(fl)
    
    img_read_ = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    image_data_gray.append(img_read_)    
    image_data.append(img_read)    
    
    #translate kinga  => 0
    #          wiktor => 1
    #label_read = os.path.split(fl)[1].split(".")[0].split('_')[0]
    image_labels.append(fl)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


xs_clahe = []
cropped = []
cropped_labels = []
cropped_names = []
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(31, 31))
for i, im in enumerate(image_data):
    x_clahe = clahe.apply(image_data_gray[i])
    xs_clahe.append(x_clahe)
    class_ = face_cascade.detectMultiScale(image_data_gray[i])
    
    for cl in class_:
        x, y, w, h = cl

        cropped.append(im[y:y+h, x:x+w])
        cropped_labels.append(image_labels[i])
        cropped_names.append(dataset_paths[i])
#
#print('nb faces:', len(cropped))
#import time
#for i, c in enumerate(cropped):
#    plt.imshow(c, cmap='gray')
#    plt.show()
#    print('file:', cropped_labels[i])
#    
#plt.imshow(cropped[5], cmap='gray')
#plt.imshow(cropped[6], cmap='gray')
#plt.imshow(cropped[7], cmap='gray')
#plt.imshow(cropped[8], cmap='gray')
#plt.imshow(cropped[9], cmap='gray')
#plt.imshow(cropped[10], cmap='gray')
#plt.imshow(cropped[11], cmap='gray')
#plt.imshow(cropped[12], cmap='gray')
#plt.imshow(cropped[13], cmap='gray')

index = 171
for i, c in enumerate(cropped):
    cv2.imwrite('./CNN_datatest_cropped/wiktor_'+str(i+index)+'.JPG', cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
#plt.imshow(x_clahe, cmap='gray')
