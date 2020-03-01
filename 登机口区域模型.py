# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:22:56 2018

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
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

#读取训练数据
dataTrain23 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace23.csv')
dataTrain22 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace22.csv')
dataTrain21 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace21.csv')
dataTrain20 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace20.csv')
dataTrain19 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace19.csv')
dataTrain18 = pd.read_csv(r'./data/dataFeatureOfflineTrainSpace18.csv')
dataTrain = pd.concat([dataTrain23,dataTrain22,dataTrain21,dataTrain20,dataTrain19,dataTrain18])
#读取测试数据
dataTest = pd.read_csv(r'./data/dataFeatureOfflineTestSpace.csv')

print("dataTrain"+str(dataTrain.shape))
print("dataTest"+str(dataTest.shape))
#选取特征构建模型
features = [ '-1minutes', '1minutes', '2minutes',
       '3minutes', '4minutes', '5minutes', '6minutes', 'hour',
       'sect', '-1minutes_',
       '1minutes_', '2minutes_', '3minutes_', '4minutes_', '5minutes_',
       '6minutes_','passengerCount7mean','passengerCount7median']
modelTrain = dataTrain.drop(['WIFIAPTag','Wifi_Obersve','time','date','passengerCount'],axis=1)
modelTest = dataTest.drop(['WIFIAPTag','Wifi_Obersve','time','date','passengerCount'],axis=1)
modelTrain = modelTrain[features]
modelTest = modelTest[features]
#
modelID = dataTrain['passengerCount']-dataTrain['passengerCount7mean']
#
#初始化LGB模型
LGB = LGBMRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8
        )

#LGB =  Ridge(alpha=1.0)

model = LGB.fit(modelTrain,modelID)
predict = model.predict(modelTest)      
dataTest['predict'] = predict
dataTest['predict'] = dataTest['predict']+dataTest['passengerCount7mean']       
dataTest['predict'] = [max(0,x) for x in dataTest['predict'] ]
#dataTest[['WIFIAPTag','hour','sect','predict']].to_csv('dataTestPort.csv',index=False)
dataTest['score'] = (dataTest['predict'] - dataTest['passengerCount'])**2
dataTest.to_csv('./data/dataTestAir.csv',index=False)        
score = dataTest['score'].sum()
print(score)
#分区域统计
dataTest['careImportance'] = [x.split('-')[0] for x in dataTest['WIFIAPTag']]
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = dataTest.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(test_div)

#登机口区域模型存储
dataAllBoo = pd.read_csv('dataTestboosting.csv')

dataAllBoo = pd.merge(dataTest,dataAllBoo,on=['WIFIAPTag','hour','sect','passengerCount'])
dataAllBoo['score'] = (dataAllBoo['pre'] - dataAllBoo['passengerCount'])**2
scorePre = dataAllBoo['score'].sum()
print(scorePre)
dataAllBoo['careImportance'] = [x.split('-')[0] for x in dataAllBoo['WIFIAPTag']]
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = dataAllBoo.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(test_div)


#登机口模型融合系统性能
dataAllBoo['pre'] = dataAllBoo['pre']*0.5 + dataAllBoo['predict']*0.5
dataAllBoo['score'] = (dataAllBoo['pre'] - dataAllBoo['passengerCount'])**2
scorePre = dataAllBoo['score'].sum()
print(scorePre)
test_div = dataAllBoo.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(test_div)


