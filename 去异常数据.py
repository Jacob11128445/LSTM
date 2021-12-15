import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from pandas import read_csv
path="C:/Users/X1Yoga2018/Desktop/LSTM"
data = read_csv('1275000519(1).csv', engine='python', skipfooter=3)
data2=data.drop('cid', axis=1)
data2.values
data2 = data2.fillna(0)
col = list(data2.columns) 
index = list(data2.index) 
data2.describe().to_csv('rest_describe.csv') 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.figure() 
data2.boxplot(rot=50) 
plt.xlabel('Time') 
plt.ylabel('Power load') 
name = list(data2.columns) 
name = name[0:len(name)-1] 
#plt.show() 


x_mean = list(data2.mean()) 
for i in range(0, len(data2.columns)): 
 Q1 = np.percentile(data2[col[i]], 25) 
 Q3 = np.percentile(data2[col[i]], 75) 
 IQR = Q3-Q1 
 for j in range(len(index)): 
    if data2[col[i]][index[j]] >= 1.5 * IQR + Q3 or data2[col[i]][index[j]] < Q1-1.5*IQR: 
        data2[col[i]][index[j]] = x_mean[i]
plt.figure() 
data2.boxplot(rot=50) 
plt.xlabel('Time') 
plt.ylabel('Power load') 
plt.show() 
data2.to_csv('去异数据.csv', encoding='UTF-8') 
