import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df=pd.read_csv('DIS.csv')
print(df.head())

def create_dataset(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)

np.random.seed(7)
all_y=df['Close'].values
dataset=all_y.reshape(-1,1)
#Normalising the data
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)-train_size
train_size,test_size=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

look_back=10
trainX,trainY=create_dataset(train_size,look_back)
testX,testY=create_dataset(test_size,look_back)

trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1))

model=Sequential()
model.add(LSTM(25,input_shape=(look_back,1)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=1,batch_size=1,verbose=2)

trainPredict=model.predict(trainX)
testPredict=model.predict(testX)

trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform([trainY])
testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot=np.empty_like(dataset)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:]=trainPredict

testPredictPlot=np.empty_like(dataset)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1,:]=testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
