from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from  random import *
import math
import matplotlib.pyplot as plt 
from itertools import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
First_Dealing = True
#数据预处理，处理时间戳，天，小时，分钟，十分钟
if First_Dealing:
    airport_gz_WiFi_ap = pd.read_csv('.\data\WIFI_AP_Passenger_Records_chusai_2ndround.csv')
    
    airport_gz_WiFi_ap['date'] = [x.split('-')[0]+"-"+x.split('-')[1]+"-"+x.split('-')[2] for x in airport_gz_WiFi_ap['timeStamp']]
    airport_gz_WiFi_ap['hour'] = [x.split('-')[3] for x in airport_gz_WiFi_ap['timeStamp']]
    airport_gz_WiFi_ap['minutes'] = [x.split('-')[4] for x in airport_gz_WiFi_ap['timeStamp']]
    
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
    airport_gz_WiFi_ap['sect'] = [couterCal(int(x)) for x in airport_gz_WiFi_ap['minutes']]      
    
    
    
    dataOfflinePreUsed = airport_gz_WiFi_ap.groupby(by=['WIFIAPTag','date','hour','sect'],as_index=False)['passengerCount'].mean()
    dataOfflinePreUsed=dataOfflinePreUsed.sort_index(by=['WIFIAPTag','date','hour','sect'])
    
    
    dataOfflinePreUsed.to_csv(r'.\data\airport_gz_WiFi_apAfter.csv',index=False)


