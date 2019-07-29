#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:56:10 2019

@author: mahbubcseju
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

training_data=pd.read_csv("/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/RNN/Google_Stock_Price_Train.csv")
train_data=training_data.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
scaled_train_data=sc.fit_transform(train_data)

x_train=[]
y_train=[]
for i in range(60,len(scaled_train_data)):
    x_train.append(scaled_train_data[i-60:i,0])
    y_train.append(scaled_train_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

#reshapping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor =Sequential()
#units is number of neuron
regressor.add(LSTM(units = 50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))
#Second LSTM layer
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))
#Third LSTM layer
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))
#If last layer then no sequence
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer="adam",loss="mean_squared_error")
regressor.fit(x_train,y_train,epochs=100,batch_size=32)

testing_data=pd.read_csv("/home/mahbubcseju/Desktop/Anaconda/Machine-Learning-Practice/RNN/Google_Stock_Price_Test.csv")
test_data=testing_data.iloc[:,1:2].values
total_data=pd.concat((training_data['Open'],testing_data['Open']),axis=0)
total_test_data=total_data[len(training_data)-60:].values
total_test_data=total_test_data.reshape(-1,1)
total_test_data=sc.transform(total_test_data)

x_test=[]
for i in range(60,len(total_test_data)):
    x_test.append(total_test_data[i-60:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

test_data_result=regressor.predict(x_test)
test_data_result=sc.inverse_transform(test_data_result)
plot.plot(test_data,color='red',label="Real Google Stock Price")
plot.plot(test_data_result,color='green',label="Predicted Google Stock Price")
plot.title("RNN implementation")
plot.xlabel('Time')
plot.ylabel('Price')
plot.legend()
plot.show()