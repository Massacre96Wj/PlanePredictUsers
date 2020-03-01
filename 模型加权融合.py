# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:56:32 2018

@author: 
"""
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from datetime import datetime 
from dateutil.parser import parse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from scipy.stats import boxcox
from lightgbm.sklearn import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

dataTest = pd.read_csv('./data/dataTestAir.csv')#登机口区域模型
dataTestBoosting = pd.read_csv('./data/dataTestAllBoosting.csv')#整体区域模型

dataTest_ = dataTest[['WIFIAPTag', 'hour', 'sect','predict']]
dataTestBoosting_ = dataTestBoosting[['WIFIAPTag', 'hour', 'sect','predict','passengerCount']]
data = pd.merge(dataTest_,dataTestBoosting_,on=['WIFIAPTag', 'hour', 'sect'],how='outer')
data['pre'] = data['predict_y']
index = data[~data['predict_x'].isnull()].index
data.ix[index,'pre'] = data.ix[index,'predict_x']*0.5+data.ix[index,'predict_y']*0.5#直接加权融合
data['score'] = data['pre'] - data['passengerCount']
data['score'] = [(x)**2 for x in data['score']]
data['careImportance'] = [x.split('-')[0] for x in data['WIFIAPTag']]#区域统计
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
data = data[data['careImportance'].isin(listCare)]

score = data['score'].sum()
data['careImportance'] = [x.split('-')[0] for x in data['WIFIAPTag']]
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = data.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(score)  
print(test_div) 

