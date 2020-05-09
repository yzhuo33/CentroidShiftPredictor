# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:34:57 2020

@author: Ya Zhuo, University of Houston
"""

# Call functions
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Load training data
DE = pd.read_excel('relative_permittivity_training_set.xlsx')
array = DE.values
X = array[:,2:100]
Y = array[:,1]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=15, shuffle=True)

# Data transformation
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# SVR model construction
SVM = SVR(kernel='rbf',C=10**1.75, epsilon=0.1, gamma=0.01).fit(X_train, Y_train)

# Prediction
prediction = pd.read_excel('to_predict_relative_permittivity.xlsx')
a = prediction.values
b = a[:,1:99]
c=scaler.transform(b)
result=SVM.predict(c)
composition=pd.read_excel('to_predict_relative_permittivity.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)
result=pd.DataFrame(result)
predicted=np.column_stack((composition,result))
predicted=pd.DataFrame(predicted)
predicted.to_excel('predicted_relative_permittivity.xlsx', index=False, header=("Composition","Predicted relative permittivity"))
print("A file named predicted_relative_permittivity.xlsx has been generated.\nPlease check your folder.")