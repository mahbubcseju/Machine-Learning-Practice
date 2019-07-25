#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:07:05 2019

@author: mahbubcseju
"""
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
from cv2 import cvtColor
import cv2


classifier = CascadeClassifier("haarcascade_frontalface_default.xml")
image = imread("test.jpg")
gray_image = cvtColor(image,cv2.COLOR_BGR2GRAY)
boxes=classifier.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=5)

for (x,y,w,h) in boxes:
    rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
imshow('img',image)
k=waitKey(0)
if k==27:
    destroyAllWindows()
