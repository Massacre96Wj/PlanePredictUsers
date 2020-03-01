# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 16:36:09 2018

@author: 
"""

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from itertools import product
from pandas.tseries.offsets import *




dataTrain= pd.read_csv(r'./data/airport_gz_WiFi_apAfter.csv')#原始数据读取

dataTrain['careImportance'] = [x.split('-')[0] for x in dataTrain['WIFIAPTag']]#大区域
listCare = ['E1','E2','E3','EC','T1','W1','W2','W3','WC']
dataTrain = dataTrain[dataTrain['careImportance'].isin(listCare)]
WIFIAPTag_list = list(dataTrain.WIFIAPTag.unique())
dataTrain = dataTrain.drop(['careImportance'],axis=1)


dataTrain['Wifi_Obersve'] = [x[0:5] for x in dataTrain['WIFIAPTag']]#小区域

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
#        dateMax = dataTrainOffline.date.max()
#        dateSplit = dateMax.split('-')
#        dateCompare = dateSplit[0]+"-"+dateSplit[1]+"-"+str(int(dateSplit[2])-i)
        dataUsed = dataTrainOffline[dataTrainOffline.date>dataTrainOffline.date.max()-i*Day()]   
#        dataUsed = dataTrainOffline[dataTrainOffline.date>dateCompare]   
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


print('-----feature-stastic------\n')

dateWindow = ["2016-09-24"]
dataFeatureOfflineUsed = []
for date in dateWindow:
    print(date)
    dataTrainUsed = dataTrain[dataTrain['date']<parse(date)]
    dataLabelUsed = dataTrain[dataTrain['date']>=parse(date)]
#    dateSplit = date.split('-')
#    dateCompare = dateSplit[0]+"-"+dateSplit[1]+"-"+str(int(dateSplit[2])+1)
    dataLabelUsed = dataLabelUsed[dataLabelUsed['date']<(parse(date)+1*Day())]
    
    date_list = list(dataLabelUsed.date.unique())
    hour_list = list(dataLabelUsed.hour.unique())
    sect_list = list(dataLabelUsed.sect.unique())
    
    date_list_ = []
    hour_list_ = []
    sect_list_ = []
    WIFIAPTag_list_ = []
    
    dataUsed = product(date_list,hour_list,sect_list,WIFIAPTag_list)
    for element in dataUsed:
        date_list_.append(element[0])
        hour_list_.append(element[1])
        sect_list_.append(element[2])
        WIFIAPTag_list_.append(element[3])
        
    df = pd.DataFrame(WIFIAPTag_list_,columns=['WIFIAPTag'])
    
    df['date'] = date_list_  
    df['hour'] = hour_list_ 
    df['sect'] = sect_list_ 
    dataLabelUsed = pd.merge(dataLabelUsed,df,on=["WIFIAPTag","date","hour","sect"],how='left').fillna(0)
    
     
    dataUsedQtyCombine = dataUsedCombine(feature_build(dataTrainUsed,[1,3,5,7,9]),[1,3,5,7,9]).fillna(0)#    
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


dataFeatureOfflineTest = pd.concat(dataFeatureOfflineUsed)
dataFeatureOfflineTest = dataFeatureOfflineTest[dataFeatureOfflineTest['date']!='1970-01-01']
#交互信息提取
dataFeatureOfflineTest['combine1']= dataFeatureOfflineTest['passengerCount3median']+dataFeatureOfflineTest['passengerCount3mean']
dataFeatureOfflineTest['combine2']= dataFeatureOfflineTest['passengerCount5median']+dataFeatureOfflineTest['passengerCount5mean']
dataFeatureOfflineTest['combine3']= dataFeatureOfflineTest['passengerCount7median']+dataFeatureOfflineTest['passengerCount7mean']
dataFeatureOfflineTest['combine4']= dataFeatureOfflineTest['passengerCount9median']+dataFeatureOfflineTest['passengerCount9mean']


dataFeatureOfflineTest['combine5']= dataFeatureOfflineTest['passengerCount_hour3median']+dataFeatureOfflineTest['passengerCount_hour3mean']
dataFeatureOfflineTest['combine6']= dataFeatureOfflineTest['passengerCount_hour5median']+dataFeatureOfflineTest['passengerCount_hour5mean']
dataFeatureOfflineTest['combine7']= dataFeatureOfflineTest['passengerCount_hour7median']+dataFeatureOfflineTest['passengerCount_hour7mean']
dataFeatureOfflineTest['combine8']= dataFeatureOfflineTest['passengerCount_hour9median']+dataFeatureOfflineTest['passengerCount_hour9mean']


dataFeatureOfflineTest['combine9']= dataFeatureOfflineTest['passengerCountObersve3mean']+dataFeatureOfflineTest['passengerCountObersve3median']
dataFeatureOfflineTest['combine10']= dataFeatureOfflineTest['passengerCountObersve5mean']+dataFeatureOfflineTest['passengerCountObersve5median']
dataFeatureOfflineTest['combine11']= dataFeatureOfflineTest['passengerCountObersve7mean']+dataFeatureOfflineTest['passengerCountObersve7median']
dataFeatureOfflineTest['combine12']= dataFeatureOfflineTest['passengerCountObersve9mean']+dataFeatureOfflineTest['passengerCountObersve9median']


dataFeatureOfflineTest['diff1']= dataFeatureOfflineTest['passengerCount3median']-dataFeatureOfflineTest['passengerCount1mean']
dataFeatureOfflineTest['diff2']= dataFeatureOfflineTest['passengerCount5median']-dataFeatureOfflineTest['passengerCount3median']
dataFeatureOfflineTest['diff3']= dataFeatureOfflineTest['passengerCount7median']-dataFeatureOfflineTest['passengerCount5median']
dataFeatureOfflineTest['diff4']= dataFeatureOfflineTest['passengerCount9median']-dataFeatureOfflineTest['passengerCount7median']
#数据存取
dataFeatureOfflineTest.to_csv(r'./data/dataFeatureOfflineTest.csv',index=False)

    


