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
from pandas.tseries.offsets import Day,Hour,Second,Minute
from sklearn.preprocessing import OneHotEncoder
from itertools import product
airport_gz_gates = pd.read_csv(r'.\data\airport_gz_gates.csv')#登机口数据

airport_gz_flights = pd.read_csv(r'.\data\airport_gz_flights_chusai_2ndround.csv')
airport_gz_flights['scheduled_flt_time'] = [parse(x)+8*Hour() for x in airport_gz_flights['scheduled_flt_time']]
airport_gz_flights = airport_gz_flights[['flight_ID','scheduled_flt_time','BGATE_ID']]
airport_gz_flights['date'] = [x.strftime('%F') for x in airport_gz_flights['scheduled_flt_time']]
airport_gz_flights['hour'] = [x.strftime('%H') for x in airport_gz_flights['scheduled_flt_time']]
airport_gz_flights['minutes'] = [x.strftime('%M') for x in airport_gz_flights['scheduled_flt_time']]
    

coor_realation = pd.read_csv(r'.\data\coor_realation.csv')
coor_realation = coor_realation.rename(columns={'wifi_ap_tag':'WIFIAPTag','bgate_id':'BGATE_ID'})
def couterCal(minutes):
    if minutes <10:
        section=0
    elif (minutes <20) and (minutes >=10):
        section=1
    elif (minutes <30) and (minutes >=20):
        section=2
    elif (minutes <40) and (minutes >=30):
        section=3
    elif (minutes <50) and (minutes >=40):
        section=4
    elif (minutes >=50):
        section=5
    return section
airport_gz_flights['sect'] = [couterCal(int(x)) for x in airport_gz_flights['minutes']]      
airport_gz_flights['time']=[parse(str(i)+" "+str(j)+":"+str(k*10)) for i,j,k in zip(airport_gz_flights['date'],airport_gz_flights['hour'],airport_gz_flights['sect'])]

airport_gz_flights = pd.merge(airport_gz_gates,airport_gz_flights,on=['BGATE_ID'],how='left')
airport_gz_flights = pd.merge(coor_realation,airport_gz_flights,on=['BGATE_ID'],how='left')
airport_gz_flights_drop = airport_gz_flights.dropna()
airport_gz_flights.to_csv(r'.\data\airport_gz_flightsAfter.csv',index=False)
#第一步去除Naz值
flights_drop_null = airport_gz_flights.dropna()
#求航班数目的时候进行去重，因为多个航班同用一个飞机
flights_drop_null = flights_drop_null.drop_duplicates(['scheduled_flt_time','BGATE_ID'])



dataTrain= pd.read_csv(r'./data/airport_gz_WiFi_apAfter.csv')
print('-----dateFormat-dealing------\n')
WIFIAPTag_list = list(coor_realation.WIFIAPTag.unique())
dataTrain = dataTrain[dataTrain['WIFIAPTag'].isin(WIFIAPTag_list)]
dataTrain['date'] = [parse(x) for x in dataTrain['date']]
dataTrain['time']=[parse(str(i)+" "+str(j)+":"+str(k*10)) for i,j,k in zip(dataTrain['date'],dataTrain['hour'],dataTrain['sect'])]
dataTrain['Wifi_Obersve'] = [x[0:5] for x in dataTrain['WIFIAPTag']]
dataTrain['weekday'] = [x.strftime("%w") for x in dataTrain['date']]
spaceTimeCount = pd.read_csv('spaceTimeCount.csv')

def nameChange(df, nameList,dic):
    for i in nameList:
        df.rename(columns={i:dic+i},inplace=True)
    return df 

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

    
def dataUsedCombine(df,dateList):
    dataTemp = df[0]
    for i in range(len(dateList)):
        if i > 0:
           dataTemp = pd.merge(dataTemp,df[i],how='outer',on=["WIFIAPTag","hour","sect"]) 
    return dataTemp


def flightsUsed(data,WIFIAPTag_list,dateRange,time):
    data_res = []
    for i,wifi in enumerate(WIFIAPTag_list):
        wifiDf = pd.DataFrame([wifi],columns=['WIFIAPTag'])
        wifiDf['time'] = time
        temp =  data[data['WIFIAPTag']==wifi].copy()
        for win in dateRange:
            wifiDf[str(win)+"minutes"] = 0  
            if win<0:
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=compare] 
                temp_ = temp_[temp_['scheduled_flt_time']<=time] 
                count = len(temp_)
                wifiDf[str(win)+"minutes"] = count  
            else:
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=time] 
                temp_ = temp_[temp_['scheduled_flt_time']<=compare] 
                count = len(temp_)
                wifiDf[str(win)+"minutes"] = count  
        data_res.append(wifiDf.copy()) 
    res = pd.concat(data_res)    
    res = res.reset_index().drop(['index'],axis=1)      
    return res

def flightsCount(data,WIFIAPTag_list,dateRange,time,spaceData):
    data_res = []
    for i,wifi in enumerate(WIFIAPTag_list):
        wifiDf = pd.DataFrame([wifi],columns=['WIFIAPTag'])
        wifiDf['time'] = time
        temp =  data[data['WIFIAPTag']==wifi].copy()
        for win in dateRange:
            wifiDf[str(win)+"minutes_"] = 0 
            if win<0:
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=compare] 
                temp_ = temp_[temp_['scheduled_flt_time']<=time] 
                temp_['diffUsed'] = [ np.round((x-time).seconds/600) for x in temp_['security_time']]
                temp_ = pd.merge(temp_,spaceTimeCount,on=['flight_ID'],how='left').fillna(0)
                temp_ = pd.merge(temp_,spaceData,on=['diffUsed'],how='left')
            else:
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=time] 
                temp_ = temp_[temp_['scheduled_flt_time']<=compare] 
                temp_['diffUsed'] = [ np.round((x-time).seconds/600) for x in temp_['security_time']]
                temp_ = pd.merge(temp_,spaceTimeCount,on=['flight_ID'],how='left').fillna(0)
                temp_ = pd.merge(temp_,spaceData,on=['diffUsed'],how='left')
        count = np.sum(temp_['sum_Times']*temp_['passenger_ID'])
        
        wifiDf[str(win)+"minutes_"] = count  
        data_res.append(wifiDf.copy()) 
    res = pd.concat(data_res)    
    res = res.reset_index().drop(['index'],axis=1)     
    return res

def flightsCountUsed(data,WIFIAPTag_list,dateRange,time,spaceTemp):
    data_res = []
    for i,wifi in enumerate(WIFIAPTag_list):
        wifiDf = pd.DataFrame([wifi],columns=['WIFIAPTag'])
        wifiDf['time'] = time
        temp =  data[data['WIFIAPTag']==wifi].copy()
        for j,win in enumerate(dateRange):
            wifiDf[str(win)+"minutes_"] = 0 
            if win<0:
                space = pd.DataFrame(spaceTemp[j],columns=['sum_Times'])
                space['diffUsed'] = [0.0,-1.0,-2.0,-3.0,-4.0,-5.0]
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=compare] 
                temp_ = temp_[temp_['scheduled_flt_time']<time] 
                temp_['diffUsed'] = [ np.round((x-time).seconds/600) for x in temp_['time']]
                temp_ = pd.merge(temp_,spaceTimeCount,on=['flight_ID'],how='left').fillna(0)
                temp_ = pd.merge(temp_,space,on=['diffUsed'],how='left')
            else:
                space = pd.DataFrame(spaceTemp[j],columns=['sum_Times'])
                space['diffUsed'] = list(range(len(spaceTemp[j])))
                compare = time+30*win*Minute()
                temp_ = temp[temp['scheduled_flt_time']>=time] 
                temp_ = temp_[temp_['scheduled_flt_time']<compare] 
                temp_['diffUsed'] = [ np.round((x-time).seconds/600) for x in temp_['time']]
                temp_ = pd.merge(temp_,spaceTimeCount,on=['flight_ID'],how='left').fillna(0)
                temp_ = pd.merge(temp_,space,on=['diffUsed'],how='left')
            count = np.sum(temp_['sum_Times']*temp_['passenger_ID'])
            wifiDf[str(win)+"minutes_"] = count  
        data_res.append(wifiDf.copy()) 
    res = pd.concat(data_res)    
    res = res.reset_index().drop(['index'],axis=1)     
    return res


def spaceTime(time,step):
    step = 10
    data = pd.read_csv('spaceTimeInformation.csv')
    data['security_time'] = [parse(x) for x in data['security_time']]
    data = data[data['security_time']<time]
    data['diffUsed'] = [np.round(x*60/step) for x in data['diff']]
    dataGroup = data.groupby(by=['diffUsed'],as_index=False)['passenger_ID'].count()
    del data
    dataGroup['prob']= dataGroup['passenger_ID']/(sum(dataGroup['passenger_ID']))
    dataGroup = dataGroup.sort_values(by=['diffUsed'],ascending=False)
    dataGroup['sum_Times']=dataGroup['prob'].cumsum() 
#    dataGroup.set_index(['diffUsed'])[['sum_Times']].plot()
    return dataGroup
#--
print('-----feature-stastic------\n')
dateWindow = ["2016-09-18"]
dataFeatureOfflineUsed = []
for date in dateWindow:
    print(date)
#    spaceData = spaceTime(parse(date),10)#空时影响
    spaceData = pd.read_csv('spaceData.csv')

    dataTrainUsed = dataTrain[dataTrain['date']<parse(date)]
    dataLabelUsed = dataTrain[dataTrain['date']>=parse(date)]

    dataLabelUsed = dataLabelUsed[dataLabelUsed['date']<(parse(date)+1*Day())]
    
    date_list = list(dataLabelUsed.date.unique())
    hour_list = list(dataLabelUsed.hour.unique())
    sect_list = list(dataLabelUsed.sect.unique())
  
    date_list_ = []
    hour_list_ = []
    sect_list_ = []
    dataUsed = product(date_list,hour_list,sect_list)
    for element in dataUsed:
        date_list_.append(element[0])
        hour_list_.append(element[1])
        sect_list_.append(element[2])
        
    time = pd.DataFrame(date_list_,columns=['date'])
    time['hour'] = hour_list_ 
    time['sect'] = sect_list_ 
    time['time']=[parse(str(i)+" "+str(j)+":"+str(k*10)) for i,j,k in zip(time['date'],time['hour'],time['sect'])]
    time = time.sort_values(by=['time'])

    time_apart = []
    for clock in time['time']:
        print(clock)
        temp_res = flightsUsed(flights_drop_null,WIFIAPTag_list,[-1,1,2,3,4,5,6],clock)
        time_apart.append(temp_res)
    flights_ = pd.concat(time_apart)
    
#    aft30 [0.6,0.5,0.4]
#    aft60 [0.6,0.5,0.4,0.3,0.2,0.1]
#    pre30 [0.2,0.5,1]
#    pre60 [0.2,0.5,1,1,0.9,0.8]
#    pre90 [0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45]
#    pre120 [0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25]
#    pre150 [0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25,0.20,0.16,0.08] 
#    pre180 [0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25,0.20,0.16,0.08,0.06,0.03,0.01] 

    prelist = [[0.6,0.5,0.4,0.3,0.2,0.1],[0.2,0.5,1],[0.2,0.5,1,1,0.9,0.8],[0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45],[0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25],[0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25,0.20,0.16,0.08] ,[0.2,0.5,1,1,0.9,0.8,0.67,0.55,0.45,0.37,0.30,0.25,0.20,0.16,0.08,0.06,0.03,0.01]]
    time_apart_ = []
    for clock in time['time']:
        print(clock)
        flightsCountUsed(airport_gz_flights_drop,['E1-3A<E1-3-03>'],[-1,1,2,3,4,5,6],clock,prelist)
        
        temp_res_ = flightsCountUsed(airport_gz_flights_drop,WIFIAPTag_list,[-1,1,2,3,4,5,6],clock,prelist)
        time_apart_.append(temp_res_)
    flights_sum = pd.concat(time_apart_)    
    dataFeatureOffline = pd.merge(flights_,dataLabelUsed,how='outer',on=["WIFIAPTag","time"]).fillna(0)
    dataFeatureOffline = pd.merge(dataFeatureOffline,flights_sum,how='outer',on=["WIFIAPTag","time"]).fillna(0)


    dataUsedQtyCombine = dataUsedCombine(feature_build(dataTrainUsed,[7]),[7]).fillna(0)#    
    dataFeatureOffline = pd.merge(dataFeatureOffline,dataUsedQtyCombine,how='outer',on=["WIFIAPTag","hour","sect"]).fillna(0)
    dataMerge = dataFeatureOffline
    dataFeatureOfflineUsed.append(dataMerge)

dataFeatureOfflineTrain = pd.concat(dataFeatureOfflineUsed)
dataFeatureOfflineTrain = dataFeatureOfflineTrain[dataFeatureOfflineTrain['date']!='1970-01-01']

dataFeatureOfflineTrain.to_csv(r'./data/dataFeatureOfflineTrainSpace18.csv',index=False)

