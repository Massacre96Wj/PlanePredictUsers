# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 16:36:09 2018

@author: 
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from itertools import *
from pandas.tseries.offsets import *
from sklearn.preprocessing import OneHotEncoder


dataTrain= pd.read_csv(r'./data/airport_gz_WiFi_apAfter.csv')#原始数据读取
dataTrain['Wifi_Obersve'] = [x[0:5] for x in dataTrain['WIFIAPTag']]#区域
print('-----dateFormat-dealing------\n')
dataTrain['date'] = [parse(x) for x in dataTrain['date']]#日期
dataTrain['weekday'] = [x.strftime("%w") for x in dataTrain['date']]#星期

#模式化列名修改
def nameChange(df, nameList,dic):
    for i in nameList:
        df.rename(columns={i:dic+i},inplace=True)
    return df 
#十分钟为粒度信息特征提取
def feature_build(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
   
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]   
        if i > 1:
            dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","hour","sect"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean','median','std','sum']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCount'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
        else:
            dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","hour","sect"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCount'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine
#星期特征提取
def feature_build_weekday(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]   
        dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","weekday","hour","sect"])
        dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
        fuctionList = ['mean']
        dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountWeek'+str(i))
        dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine
#星期以小时为单位特征提取
def feature_build_weekday_with_hour(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]   
        dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","weekday","hour"])
        dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
        fuctionList = ['mean','median','std','sum']
        dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountWeekHour'+str(i))
        dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine
#小时信息提取
def feature_build_with_hour(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]   
        if i > 1:
            dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","hour"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean','median','std','sum']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCount_hour'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
        else:
            dataUsedGroupby = dataUsed.groupby(["WIFIAPTag","hour"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCount_hour'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine

#分区域特征提取
def feature_build_Obersve(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]     
        if i > 1:
            dataUsedGroupby = dataUsed.groupby(["Wifi_Obersve","hour","sect"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean','median','std','sum']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountObersve'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
        else:
            dataUsedGroupby = dataUsed.groupby(["Wifi_Obersve","hour","sect"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountObersve'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine
#分区域以小时为单位特征提取
def feature_build_Obersve_hour(dataTrainOffline,dataRange):
    dataUsedQtyCombine = []
    for i in dataRange:
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]     
        if i > 1:
            dataUsedGroupby = dataUsed.groupby(["Wifi_Obersve","hour"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean','median','std','sum']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountObersveHour'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
        else:
            dataUsedGroupby = dataUsed.groupby(["Wifi_Obersve","hour"])
            dataUsedGroupbyQty = dataUsedGroupby['passengerCount']
            fuctionList = ['mean']
            dataUsedQty = nameChange(dataUsedGroupbyQty.agg(fuctionList).reset_index(),fuctionList,'passengerCountObersveHour'+str(i))
            dataUsedQtyCombine.append(dataUsedQty)
    return dataUsedQtyCombine
#模式化特征融合      
def dataUsedCombine(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["WIFIAPTag","hour","sect"]) 
    return dataTemp

def dataUsedCombineWeekday(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["WIFIAPTag","weekday","hour","sect"]) 
    return dataTemp

def dataUsedCombineWeekdayHour(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["WIFIAPTag","weekday","hour"]) 
    return dataTemp

def dataUsedCombineHour(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["WIFIAPTag","hour"]) 
    return dataTemp
    
def dataUsedCombineObserve(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["Wifi_Obersve","hour","sect"]) 
    return dataTemp
    
def dataUsedCombineObserveHour(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["Wifi_Obersve","hour"]) 
    return dataTemp
#one-hot-enconder
print('-----one-hot-enconder------\n')

#--
print('-----feature-stastic------\n')
dateWindow = ["2016-09-23","2016-09-22","2016-09-21"]
dataFeatureOfflineUsed = []
for date in dateWindow:
    print(date)
    dataTrainUsed = dataTrain[dataTrain['date']<parse(date)]
    
    dataLabelUsed = dataTrain[dataTrain['date']>=parse(date)]
#    dateSplit = date.split('-')
#    dateCompare = dateSplit[0]+"-"+dateSplit[1]+"-"+str(int(dateSplit[2])+1)
#    dataLabelUsed = dataLabelUsed[dataLabelUsed['date']<dateCompare]
    dataLabelUsed = dataLabelUsed[dataLabelUsed['date']<(parse(date)+1*Day())]
    
    dataUsedQtyCombine = dataUsedCombine(feature_build(dataTrainUsed,[1,3,5,7,9]),[1,3,5,7,9]).fillna(0)
    dataUsedQtyCombineHour = dataUsedCombineHour(feature_build_with_hour(dataTrainUsed,[1,3,5,7,9]),[1,3,5,7,9]).fillna(0)#  
    dataUsedQtyCombineOberse = dataUsedCombineObserve(feature_build_Obersve(dataTrainUsed,[1,3,5,7,9]),[1,3,5,7,9]).fillna(0)#  
    dataUsedQtyCombineOberseHour = dataUsedCombineObserveHour(feature_build_Obersve_hour(dataTrainUsed,[1,3,5,7,9]),[1,3,5,7,9]).fillna(0)#  

    dataUsedQtyCombineWeekday = dataUsedCombineWeekday(feature_build_weekday(dataTrainUsed,[7,14]),[7,14]).fillna(0)#                                                    
    dataUsedQtyCombineWeekdayHour = dataUsedCombineWeekdayHour(feature_build_weekday_with_hour(dataTrainUsed,[7,14]),[7,14]).fillna(0)#                                                    


    dataUsedQtyCombine = pd.merge(dataUsedQtyCombine,dataUsedQtyCombineHour,on=['WIFIAPTag','hour'],how='outer').fillna(0)
   
    dataFeatureOffline = pd.merge(dataUsedQtyCombine,dataLabelUsed,how='outer',on=["WIFIAPTag","hour","sect"]).fillna(0)
    dataFeatureOffline = pd.merge(dataFeatureOffline,dataUsedQtyCombineOberse,on=['Wifi_Obersve','hour',"sect"],how='outer').fillna(0)
    dataFeatureOffline = pd.merge(dataFeatureOffline,dataUsedQtyCombineOberseHour,on=['Wifi_Obersve','hour'],how='outer').fillna(0)
    dataFeatureOffline = pd.merge(dataFeatureOffline,dataUsedQtyCombineWeekday,on=['WIFIAPTag',"weekday",'hour',"sect"],how='outer').fillna(0)
    dataFeatureOffline = pd.merge(dataFeatureOffline,dataUsedQtyCombineWeekdayHour,on=['WIFIAPTag',"weekday",'hour'],how='outer').fillna(0)


    dataMerge = dataFeatureOffline
    dataFeatureOfflineUsed.append(dataMerge)

dataFeatureOfflineTrain = pd.concat(dataFeatureOfflineUsed)
dataFeatureOfflineTrain = dataFeatureOfflineTrain[dataFeatureOfflineTrain['date']!='1970-01-01']
#交互信息提取
#---featureCombine
dataFeatureOfflineTrain['combine1']= dataFeatureOfflineTrain['passengerCount3median']+dataFeatureOfflineTrain['passengerCount3mean']
dataFeatureOfflineTrain['combine2']= dataFeatureOfflineTrain['passengerCount5median']+dataFeatureOfflineTrain['passengerCount5mean']
dataFeatureOfflineTrain['combine3']= dataFeatureOfflineTrain['passengerCount7median']+dataFeatureOfflineTrain['passengerCount7mean']
dataFeatureOfflineTrain['combine4']= dataFeatureOfflineTrain['passengerCount9median']+dataFeatureOfflineTrain['passengerCount9mean']

dataFeatureOfflineTrain['combine5']= dataFeatureOfflineTrain['passengerCount_hour3median']+dataFeatureOfflineTrain['passengerCount_hour3mean']
dataFeatureOfflineTrain['combine6']= dataFeatureOfflineTrain['passengerCount_hour5median']+dataFeatureOfflineTrain['passengerCount_hour5mean']
dataFeatureOfflineTrain['combine7']= dataFeatureOfflineTrain['passengerCount_hour7median']+dataFeatureOfflineTrain['passengerCount_hour7mean']
dataFeatureOfflineTrain['combine8']= dataFeatureOfflineTrain['passengerCount_hour9median']+dataFeatureOfflineTrain['passengerCount_hour9mean']


dataFeatureOfflineTrain['combine9']= dataFeatureOfflineTrain['passengerCountObersve3mean']+dataFeatureOfflineTrain['passengerCountObersve3median']
dataFeatureOfflineTrain['combine10']= dataFeatureOfflineTrain['passengerCountObersve5mean']+dataFeatureOfflineTrain['passengerCountObersve5median']
dataFeatureOfflineTrain['combine11']= dataFeatureOfflineTrain['passengerCountObersve7mean']+dataFeatureOfflineTrain['passengerCountObersve7median']
dataFeatureOfflineTrain['combine12']= dataFeatureOfflineTrain['passengerCountObersve9mean']+dataFeatureOfflineTrain['passengerCountObersve9median']


dataFeatureOfflineTrain['diff1']= dataFeatureOfflineTrain['passengerCount3median']-dataFeatureOfflineTrain['passengerCount1mean']
dataFeatureOfflineTrain['diff2']= dataFeatureOfflineTrain['passengerCount5median']-dataFeatureOfflineTrain['passengerCount3median']
dataFeatureOfflineTrain['diff3']= dataFeatureOfflineTrain['passengerCount7median']-dataFeatureOfflineTrain['passengerCount5median']
dataFeatureOfflineTrain['diff4']= dataFeatureOfflineTrain['passengerCount9median']-dataFeatureOfflineTrain['passengerCount7median']
#

dataFeatureOfflineTrain.to_csv(r'./data/dataFeatureOfflineTrain.csv',index=False)
            
