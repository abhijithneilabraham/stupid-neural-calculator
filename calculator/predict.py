import numpy as np
from tensorflow.keras.models import load_model
from sklearn.externals import joblib 
scaler_y = joblib.load('transform.pkl') 
x1=int(input("enter first number="))
x2=int(input("enter second number="))
pred_data=[[x1,x2]]
pred_data=np.asarray(list(map(np.asarray,pred_data)))
# print(pred_data,dataset)
# print(x,y)
model=load_model("addition.h5")
pred=model.predict(pred_data)
ynew = scaler_y.inverse_transform(pred)
print("predicted=",ynew[0][0])