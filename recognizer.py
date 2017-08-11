#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:32:09 2017

@author: wrutka
"""

import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

import numpy as np


class FaceDetector(object):
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(31, 31))
        
    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_clahe = self.clahe.apply(img)
        class_ = self.face_cascade.detectMultiScale(x_clahe, 1.3, 5)
        
        return class_
    
class FaceRecogniser(object):
    
    def __init__(self):
        self.classifier = load_model('FR_model_ann_layers_4_cnn_layers_4-acc_1.000000_loss_0.001747_val_acc_1.000000_val_loss_0.008827.hdf5')
        self.class_mapper = {
                0: 'Ludovic',
                1: 'Wiktor',
                }

    def who_am_i(self, img):
#        img = image.load_img(file, target_size=(200, 200))
        img = cv2.resize(img, (200, 200))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        res = self.classifier.predict_classes(img)
        return self.class_mapper[res[0][0]]
            


detector = FaceDetector()
recognize = FaceRecogniser()
v = cv2.VideoCapture(0)
success, img = v.read()


#i = 0
#ii = 45
while success:
#    v.set(2, 10*i)
#    v.grab()
# Capture frame-by-frame
    ret, frame = v.read()
    faces_coords = detector.detect(frame)
    # Our operations on the frame come here
    for coords in faces_coords:
        x, y, w,h = coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        face = frame[y:y+h, x:x+w]
        
        cv2.putText(frame, recognize.who_am_i(face), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), thickness=4)
#        plt.imshow(face, cmap='rgb')
#        plt.show()
    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
v.release()
cv2.destroyAllWindows()
#    success, img = v.read()
#    if not i%100:
#        cv2.imwrite('./new_wiktor/wiktor_'+str(ii)+'.JPG', img)
#        ii+=1
#    i+=10
    
v.release() 