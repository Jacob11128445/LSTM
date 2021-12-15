import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pylab
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error
import os;
path="C:/Users/X1Yoga2018/Desktop/LSTM"
os.chdir(path)
os.getcwd()
np.random.seed(2021)
def create_dataset(dataset, previous=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-previous-1):
        a = dataset[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(dataset[i + previous, 0])
    return np.array(dataX), np.array(dataY)
np.random.seed(2021)    
################################################################
'''
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
'''


df = read_csv('去异数据.csv', engine='python', skipfooter=3)
df2=df.drop('cid', axis=1)
df2 = df2.fillna(0.01)
df3=df2.values
dataset=np.sum(df3, axis=1, dtype=float)

dataset

from numpy import log
dataset = log(dataset)

meankwh=np.mean(dataset)
# test split data

# normalize dataset with MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = dataset.reshape(-1,1)
dataset = scaler.fit_transform(dataset)
print(dataset)


# Training and Test data partition
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t-50 and Y=t
previous = 50
X_train, Y_train = create_dataset(train, previous)
X_test, Y_test = create_dataset(test, previous)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# LSTM的产生和预测
# 对模型进行了100多个时期的训练，并生成了预测。
model = Sequential()
#model.add(LSTM(4, input_shape=(1, previous)))
model.add(LSTM(50, activation='relu', input_shape=(1, previous))) 
model.add(RepeatVector(1)) 
model.add(LSTM(20, activation='relu', return_sequences=True)) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

trainpred = model.predict(X_train)
trainpred = trainpred.reshape(-1,1)

testpred = model.predict(X_test)
testpred = testpred.reshape(-1,1)
###ss = StandardScaler()
###res_data = ss.fit_transform(res_data)
###res_data = ss.inverse_transform(res_data)
trainpred = scaler.inverse_transform(trainpred)
Y_train = scaler.inverse_transform([Y_train])
testpred = scaler.inverse_transform(testpred)
Y_test = scaler.inverse_transform([Y_test])
predictions = testpred

trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainpredPlot = np.empty_like(dataset)
trainpredPlot[:, :] = np.nan
trainpredPlot[previous:len(trainpred)+previous, :] = trainpred

testpredPlot = np.empty_like(dataset)
testpredPlot[:, :] = np.nan
testpredPlot[len(trainpred)+(previous*2)+1:len(dataset)-1, :] = testpred

inversetransform, =plt.plot(scaler.inverse_transform(dataset))
trainpred, =plt.plot(trainpredPlot)
testpred, =plt.plot(testpredPlot)
plt.title("Predicted vs. Actual Consumption")
plt.show()
