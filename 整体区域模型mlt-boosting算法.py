# -*- coding: utf-8 -*-
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from datetime import datetime 
from dateutil.parser import parse
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm.sklearn import LGBMRegressor
import autograd.numpy as anp
from sklearn import linear_model
from sklearn.cross_validation import KFold
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import boxcox
import warnings
warnings.filterwarnings("ignore")

#sigmod变换
def sigmod(x):
    y = 1/(1+np.e**(-x))
    return y
#sigmod反变换
def sigmod_trans(x):
    y = (np.log(1/(x)-1))*(-1)
    return y

def plot_method(x,name):    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(x, bins=100,color='black')
    plt.title(name)
    plt.xlabel('label')

#mlt-boosting算法
class boostingChange(object):
#模型初始化
    def __init__(self, train,test,train_label,First_change,steps,n_estimators_list):    
        self.train = train
        self.test = test
        self.train_label = train_label
        self.First_change= First_change#首步是否进行boxcox变换
        self.steps = steps#最大迭代阶段次数
        self.n_estimators_list = n_estimators_list#阶段迭代次数列表
        self.model_list=[]
        self.learning_rate_stage = 1
        self.box_value = []
        self.add_value = []
#模型训练        
    def fit(self):
        if self.First_change:#boxcox变换
            act  = boxcox(self.train_label+0.1)[0]
            self.act_ = boxcox(self.train_label+0.1)[1]
        else:
            act = self.train_label
        steps=self.steps
        actual = act
        n_samples = len(self.train_label)
        y_pred_train = np.zeros(n_samples, np.float32)
        n_estimators_list = self.n_estimators_list
        for i in range(1):
            num = np.random.randint(0,5000)
            print("----training begin----")  
            for step in range(steps):
                print(step) 
                actual = actual-y_pred_train#残差计算
                if step>0: #残差进行标签压缩变换，和boxcox变换           
                    actual_ = sigmod(actual)
                    actual_box = boxcox(actual_)[0]
                    actual_box_val = boxcox(actual_)[1]
                    self.box_value.append(actual_box_val)
                    actual_used = actual_box                    
                else:
                    actual_used = actual
				#阶段模型生成	
                model = LGBMRegressor(n_estimators=n_estimators_list[step],max_depth=3,learning_rate=0.02,subsample=1,colsample_bytree=1)
                model.fit(self.train.values,actual_used)#阶段模型训练
                y_pred_train_ = model.predict(self.train.values)#阶段预测输出
                if step>0:#阶段反变换计算输出
                    y_pred_train = (y_pred_train_*actual_box_val+1)**(1/actual_box_val)
                    y_pred_train = sigmod_trans(y_pred_train)
                else:
                    y_pred_train = y_pred_train_
                self.model_list.append(model)#阶段模型存储
#模型预测                
    def predict(self):
        print("----predicting begin----")   
        y_pred = np.zeros(self.test.shape[0], np.float32)
        for i, base_model in enumerate(self.model_list):
            print(i)
            value_turn = self.learning_rate_stage*base_model.predict(self.test.values)
            if i>0:
                box = self.box_value.pop()
                value_turn = (value_turn*box+1)**(1/box)
                value_turn = sigmod_trans(value_turn)
            y_pred += value_turn
        #---
        if self.First_change:
            y_pred = (y_pred*self.act_+1)**(1/self.act_)-0.1   
        return y_pred
    

dataTest = pd.read_csv(r'./data/dataFeatureOfflineTest.csv')#测试数据
dataTest = dataTest.fillna(0)
dataTrain = pd.read_csv(r'./data/dataFeatureOfflineTrain.csv')#训练数据
dataTrain = dataTrain.head(int(dataTrain.shape[0]*0.6))
dataTrain = dataTrain.fillna(0)

train = dataTrain.drop(['WIFIAPTag','Wifi_Obersve','date','passengerCount'],axis=1)

test = dataTest.drop(['WIFIAPTag','Wifi_Obersve','date','passengerCount'],axis=1)

train_label = dataTrain['passengerCount']
First_change=True
steps = 2
n_estimators_list = [200,200]#迭代次数列表
model = boostingChange(train,test,train_label,First_change,steps,n_estimators_list)#模型初始化
model.fit()
y_pred = model.predict()

dataTest['predict'] = [np.max([0,x]) for x in y_pred]
dataTest['score'] = (dataTest['predict'] - dataTest['passengerCount'])**2
dataTest.to_csv('./data/dataTestboosting.csv',index=False)        
score = dataTest['score'].sum()

dataTest['careImportance'] = [x.split('-')[0] for x in dataTest['WIFIAPTag']]#分区域统计
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = dataTest.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(score)  
print(test_div) 

#---------------------------------------------------------------------
#非mlt-boosting算法系统性能仿真
dataTest = pd.read_csv(r'./data/dataFeatureOfflineTest.csv')
dataTest = dataTest.fillna(0)
dataTrain = pd.read_csv(r'./data/dataFeatureOfflineTrain.csv')
dataTrain = dataTrain.fillna(0)

train = dataTrain.drop(['WIFIAPTag','Wifi_Obersve','date','passengerCount'],axis=1)

test = dataTest.drop(['WIFIAPTag','Wifi_Obersve','date','passengerCount'],axis=1)

train_label = dataTrain['passengerCount']
LGB = LGBMRegressor(n_estimators=400,max_depth=3,learning_rate=0.02,subsample=1,colsample_bytree=1)
model = LGB.fit(train,train_label)
predict = model.predict(test)
dataTest['pre'] = [np.max([0,x]) for x in predict]
dataTest['score'] = (dataTest['pre'] - dataTest['passengerCount'])**2
score = dataTest['score'].sum()
dataTest['careImportance'] = [x.split('-')[0] for x in dataTest['WIFIAPTag']]#分区域统计
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = dataTest.groupby(by=['careImportance'],as_index=False)['score'].sum()
print(score)  
print(test_div) 
print(score)
#------------------------


#基础模型系统性能仿真
dataTrain = pd.read_csv(r'./data/dataFeatureOfflineTrainPre.csv')
dataTest = pd.read_csv(r'./data/dataFeatureOfflineTestPre.csv')

modelTrain = dataTrain.drop(['WIFIAPTag','date','passengerCount'],axis=1)
modelTest = dataTest.drop(['WIFIAPTag','date','passengerCount'],axis=1)
modelID = dataTrain['passengerCount']
#


LGB=GradientBoostingRegressor(
  loss='ls'
, learning_rate=0.05#
, n_estimators=175 #
, max_depth=3
, random_state=1
, max_features = 'sqrt'
)


model = LGB.fit(modelTrain,modelID)
predict = model.predict(modelTest)
         
dataTest['pre'] = [np.max([0,x]) for x in predict]
dataTest['score'] = (dataTest['pre'] - dataTest['passengerCount'])**2
score = dataTest['score'].sum()
dataTest['careImportance'] = [x.split('-')[0] for x in dataTest['WIFIAPTag']]#分区域统计
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
test_div = dataTest.groupby(by=['careImportance'],as_index=False)['score'].sum()

print(score)  
print(test_div) 
print(score)
