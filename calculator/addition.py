import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset=[[1,1,2],[1,2,3],[3,5,8],[9,4,13],[5,7,12],[150,200,350],[100,200,300]]
dataset=np.asarray(list(map(np.asarray,dataset)))
x=dataset[:,:-1]
y=dataset[:,-1]
y=np.reshape(y, (-1,1))
# print(x,y)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x)
xscale=scaler_x.transform(x)
scaler_y.fit(y)
yscale=scaler_y.transform(y)
print(xscale,yscale)
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(xscale, yscale, epochs=10, batch_size=10,validation_split=0.2)
pred_data=[[1,1]]
pred_data=np.asarray(list(map(np.asarray,pred_data)))
# print(pred_data,dataset)
# print(x,y)
pred=model.predict(pred_data)
ynew = scaler_y.inverse_transform(pred)
print("predicted=",ynew[0][0])
