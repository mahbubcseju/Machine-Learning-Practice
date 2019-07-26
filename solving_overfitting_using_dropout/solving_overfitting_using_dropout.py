#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:03:04 2019

@author: mahbubcseju
"""

import numpy as np
import matplotlib.pyplot as matplot
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")

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
from keras.layers import Dropout

#output_dim= number of node in first hidden layer
#input_dim = number of node in input layer
#First dense will contain the number of input layer and number of hidden layer
model_builder = Sequential()
model_builder.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
model_builder.add(Dropout(p=0.1))
#second hidden layer output_dim is the number of nodes
#uniform : the wight between this node and its previous hidden layer node
model_builder.add(Dense(output_dim=6,init="uniform",activation="relu"))

model_builder.add(Dropout(p=0.1))
model_builder.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

#Optimizer is for finding the best algorithm for weight update
#Staochastic gradient Descent algorithm : adam
model_builder.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model_builder.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
prediction=model_builder.predict(standard_scaler.fit_transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
print(prediction)