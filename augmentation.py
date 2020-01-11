#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:14:35 2019

@author: mahbubcseju
"""

from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread, imsave
from skimage import exposure, color
from skimage.transform import resize


import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import os

def imgGen(img,
           zca=False,
           rotation=0.,
           w_shift=0.,
           h_shift=0.,
           shear=0.,
           zoom=0.,
           h_flip=False,
           v_flip=False,
           preprocess_fcn=None,
           batch_size=9,
           path_prefix=None,
           i=0,
           formats=None
           ):
    
    datagen = ImageDataGenerator(
            zca_whitening=zca,
            rotation_range=rotation,
            width_shift_range=w_shift,
            height_shift_range=h_shift,
            shear_range=shear,
            zoom_range=zoom,
            fill_mode='nearest',
            horizontal_flip=h_flip,
            vertical_flip=v_flip,
            preprocessing_function=preprocess_fcn,
            data_format=K.image_data_format())
    
    datagen.fit(img)

    koyta = 0
    for img_batch in datagen.flow(img, batch_size=9, shuffle=False):

        for img1 in img_batch:
            if path_prefix:
                imsave(path_prefix + "/" + str(i)+ formats,img1)
                i=i+1
                koyta += 1
                if koyta >= batch_size:
                    return i
            
        if koyta >= batch_size:
            return i
    return i
    

# Define functions for contrast adjustment

# Contrast stretching
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

# Histogram equalization
def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

# Adaptive histogram equalization
def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


preprocess_fcn = [None, contrast_stretching, HE, AHE]


image_format = ['JPEG','JPG', 'PNG']


def make_n_image(image_path, n, counter, output_directory):
    img = imread(image_path)
    ima_format = image_path.split('.')
    imsave(output_directory + "/" + str(counter) + '.' + ima_format[-1], img)
    counter += 1
    img = img.astype('float32')
    img /= 255
    h_dim = np.shape(img)[0]
    w_dim = np.shape(img)[1]
    num_channel = np.shape(img)[2]
    img = img.reshape(1, h_dim, w_dim, num_channel)

    total_aug = (n-1)
    per_aug, rest_aug = total_aug // 4, total_aug % 4

    for ind in range(4):
        fcn = preprocess_fcn[ind]
        batch_size = per_aug
        if ind < rest_aug:
            batch_size += 1
        counter = imgGen(img,
               rotation=30,
               h_shift=0.5,
               batch_size=batch_size,
               preprocess_fcn=fcn,
               path_prefix=output_directory,
               i=counter,
               formats='.' + ima_format[-1]
               )
    return counter


def process_images(image_paths, output_directory, total=500):

    total_image = min(len(image_paths), total)
    per_image = total // total_image
    rest = total % total_image
    counter = 0
    for ind in range(min(len(image_paths), total)):

        if ind < rest:
            counter = make_n_image(image_paths[ind], per_image + 1, counter, output_directory)
        else:
            counter = make_n_image(image_paths[ind], per_image, counter, output_directory)
    return counter


def augment(data_directory, augmented_directory, total):
    for directory in os.listdir(data_directory):
        sub_dir = os.path.join(data_directory, directory)
        if os.path.isdir(sub_dir):
            output_directory = os.path.join(augmented_directory, directory)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            image_paths = []
            for image in os.listdir(sub_dir):
                image_path = os.path.join(sub_dir, image)
                ima_format = image_path.split('.')
                if ima_format[-1].upper() not in image_format:
                    continue

                image_paths.append(image_path)
            process_images(image_paths, output_directory)


    for directory in os.listdir(augmented_directory):
        sub_dir = os.path.join(augmented_directory, directory)
        if os.path.isdir(sub_dir):
            print(len(os.listdir(sub_dir)))

import sys
if __name__== "__main__":

    if len(sys.argv) == 4:
        augment(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print("python augmentation.py 'data/', 'output', 500")
