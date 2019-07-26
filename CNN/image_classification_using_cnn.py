#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mahbubcseju
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def model_builder():
    classifier = Sequential()
    # Creating 32 feature detector using 3 * 3 kernal
    # input_shape : (256,256,3),3 is the number of channel for color image
    # (64,64,3) is 3 two dimensional vector of size 64 by 64
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=128, activation="relu"))
    classifier.add(Dense(output_dim=1, activation="sigmoid"))
    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifier


classifier = model_builder()


def image_preproccessing():
    from keras.preprocessing.image import ImageDataGenerator

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    train_image = train_data_gen.flow_from_directory(
        "/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/CNN/dataset/training_set", target_size=(64, 64), batch_size=32, class_mode="binary"
    )
    test_image = test_data_gen.flow_from_directory(
        "/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/CNN/dataset/test_set", target_size=(64, 64), batch_size=32, class_mode="binary"
    )
    return train_image,test_image

training_set,test_image=image_preproccessing()

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000, #Number of training image
        nb_epoch=25,
        validation_data=test_image,
        nb_val_samples=2000, #Number of test image
    )

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/CNN/dataset/single_prediction/cat_or_dog_2.jpg',
                            target_size=(64,64)
                            )
test_image =image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]>=0.51:
    prediction="dog"
else:
    prediction="cats"
