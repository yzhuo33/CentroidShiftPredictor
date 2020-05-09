# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:00:41 2020

@author: Ya Zhuo, University of Houston
"""

#import general python package/ read in compounds list
import pandas as pd 
df = pd.read_excel(r'c_pounds.xlsx')   
df.head()
df.dtypes
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
from statistics import mean
class Vectorize_Formula:

    def __init__(self):
        elem_dict = pd.read_excel(r'elements.xlsx') # CHECK NAME OF FILE 
        self.element_df = pd.DataFrame(elem_dict) 
        self.element_df.set_index('Symbol',inplace=True)
        self.column_names = []
        for string in ['avg','diff','max','min','std']:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string+'_'+column_name)

    def get_features(self, formula):
        try:
            fractional_composition = mg.Composition(formula).fractional_composition.as_dict()
            element_composition = mg.Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            std_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                except Exception as e: 
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*len(self.element_df.iloc[0])*5)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature=self.element_df.loc[list(fractional_composition.keys())].std(ddof=0)
            
            features = pd.DataFrame(np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature), np.array(std_feature)]))
            features = np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature), np.array(std_feature)])
            return features.transpose()
        except:
            print('There was an error with the Formula: '+ formula + ', this is a general exception with an unkown error')
            return [np.nan]*len(self.element_df.iloc[0])*5
gf=Vectorize_Formula()

# empty list for storage of features
features=[]

# add values to list using for loop
for formula in df['Composition']:
    features.append(gf.get_features(formula))

# feature vectors and targets as X and y 
X = pd.DataFrame(features, columns = gf.column_names)
pd.set_option('display.max_columns', None)
# allows for the export of data to excel file
header=gf.column_names
header.insert(0,"Composition")

composition=pd.read_excel('c_pounds.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)

predicted=np.column_stack((composition,X))
predicted=pd.DataFrame(predicted)
predicted.to_excel('to_predict_relative_permittivity.xlsx', index=False,header=header)
print("A file named to_predict_relative_permittivity.xlsx has been generated.\nPlease check your folder.")