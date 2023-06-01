import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from pandas_datareader import data as pdr
import yfinance as yfin
from sklearn.preprocessing import MinMaxScaler
yfin.pdr_override()
import streamlit as st
#Importing the model
from keras.layers import Dense, Dropout , LSTM
from keras.models import Sequential
#Importing the model
from keras.layers import Dense, Dropout , LSTM
from keras.models import Sequential
from keras.models import load_model
start='2010-06-30'
end='2023-02-27'
st.title('Stock Trend Prediction DL Model')
# st.subheader('AIML Project by-')
# st.subheader('Dhyanendra Tripathi(221010218)')
# st.subheader('Arpit Kumar Sinha(221010211)')
user_input=st.text_input('Enter Stock Sticker','AAPL')


df = pdr.get_data_yahoo(user_input, start, end)
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

#visualisation
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 MA and 200 MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma200 , 'r')
plt.plot(ma100 , 'g')
plt.plot(df.Close , 'b')
st.pyplot(fig)


#Splitting the data into training and testing 

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))

#SCALED DOWN THE TRAINING DATA 
data_training_array = scaler.fit_transform(data_training)

#DIVIDING THE DATA INTO XTRAIN AND YTRAIN
x_train=[]
y_train=[]
for i in range(100,data_training.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i:0])

x_train,y_train = np.array(x_train), np.array(y_train)

#compiling the model
# model=Sequential()
# model.add(LSTM(units=50,activation='relu',return_sequences = True,input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=60,activation='relu',return_sequences = True))
# model.add(Dropout(0.3))

# model.add(LSTM(units=80,activation='relu',return_sequences = True))
# model.add(Dropout(0.4))

# model.add(LSTM(units=120,activation='relu'))
# model.add(Dropout(0.5))
          
# model.add(Dense(units = 1))

# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(x_train ,y_train,epochs= 50)

# model.save('final_model.h5')

# from keras.models import load_model
if user_input=='AAPL':
 model=load_model('AAPL.h5')
elif user_input=='TSLA':
 model=load_model('TSLA.h5')
elif user_input=='^NSEI':
 model=load_model('^NSEI.h5')
elif user_input=='TTM.BA':
 model=load_model('TTM_BA.h5')
elif user_input=='RELIANCE.NS':
 model=load_model('RELIANCE_NS.h5')
elif user_input=='AMZN':
 model=load_model('AMZN.h5')
elif user_input=='TTM.BA':
 model=load_model('TTM_BA.h5')
elif user_input=='TTM.BA':
 model=load_model('TTM_BA.h5')
past_100_days=data_training.tail(100)

frames=[past_100_days,data_testing]
final_df=pd.concat(frames,ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#Making predictions
y_predicted = model.predict(x_test)

k=scaler.scale_[0]

scale_factor=1/k
y_predicted = y_predicted * scale_factor 
y_test = y_test * scale_factor

st.subheader('Predicted Price vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' ,label = 'Original Price')
plt.plot(y_predicted, 'r' ,label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
