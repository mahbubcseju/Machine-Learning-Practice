#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:03:04 2019

@author: mahbubcseju
"""

import numpy as np
import matplotlib.pyplot as matplot
import pandas as pd

dataset = pd.read_csv('/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/cross_validation_k_fold/data.csv')

x = dataset.iloc[:,3:13].values
y= dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x1=LabelEncoder()
x[:,1]=label_encoder_x1.fit_transform(x[:,1])
label_encoder_x2 = LabelEncoder()
x[:,2]=label_encoder_x2.fit_transform(x[:,2])

one_hot_encoder= OneHotEncoder(categorical_features=[1])
x=one_hot_encoder.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
standard_scaler=StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test=standard_scaler.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    #output_dim= number of node in first hidden layer
    #input_dim = number of node in input layer
    #First dense will contain the number of input layer and number of hidden layer
    model_builder = Sequential()
    model_builder.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
    #second hidden layer output_dim is the number of nodes
    #uniform : the wight between this node and its previous hidden layer node
    model_builder.add(Dense(output_dim=6,init="uniform",activation="relu"))
    model_builder.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
    
    #Optimizer is for finding the best algorithm for weight update
    #Staochastic gradient Descent algorithm : adam
    model_builder.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
    return model_builder

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn=build_classifier)
#Huyper_parameter
parameters ={
        'batch_size':[25,32],
        'nb_epoch':[250,500],
        'optimizer':['adam','rmsprop']
        }
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_