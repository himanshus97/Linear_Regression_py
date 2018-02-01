#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:30:30 2018

@author: himanshu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset = pd.read_csv("sample_data.csv") #add your data here
X=dataset.iloc[:,:-1].values #make changes here as according to your dataset
y = dataset.iloc[:,1].values

#splitting data sets
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size= 1/3,random_state =0)

#fitting data
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#prediction
y_pred = reg.predict(X_test)

#plotting training data
plt.scatter(X_train,y_train, color="red")
plt.plot(X_train,reg.predict(X_train), color ="blue")
plt.title("Salary v/s Experience (Training data)")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()

#plotting test data
plt.scatter(X_test,y_test, color="red")
plt.plot(X_train,reg.predict(X_train), color ="blue")
plt.title("Salary v/s Experience (Test data)")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()
