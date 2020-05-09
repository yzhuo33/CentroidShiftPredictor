# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:58:52 2020

@author: Ya Zhuo, University of Houston
"""

# Call functions
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load training data
DE = pd.read_excel('centroid_shift_training_set.xlsx')
array = DE.values
X = array[:,2:10]
Y = array[:,1]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,random_state=50, shuffle=True)

# XGB model construction
xgb_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.15, n_estimators=50, verbosity=1, objective='reg:squarederror',
                             booster='gbtree', tree_method='auto', n_jobs=1, gamma=0.0001, min_child_weight=8,max_delta_step=0,
                             subsample=1, colsample_bytree=1, colsample_bylevel=0.9, colsample_bynode=1, reg_alpha=0,
                             reg_lambda=4, scale_pos_weight=1, base_score=0.6, random_state=15, missing=None,
                             num_parallel_tree=1, importance_type='gain', eval_metric='rmse',nthread=4).fit(X_train,Y_train)

# Prediction
prediction = pd.read_excel('to_predict_centroid_shift.xlsx')
a = prediction.values
b = a[:,1:9]
result=xgb_model.predict(b)
composition=pd.read_excel('to_predict_centroid_shift.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)
result=pd.DataFrame(result)
predicted=np.column_stack((composition,result))
predicted=pd.DataFrame(predicted)
predicted.to_excel('predicted_centroid_shift.xlsx', index=False, header=("Composition","Predicted centroid shift"))
print("A file named predicted_centroid_shift.xlsx has been generated.\nPlease check your folder.")