import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('Churn_Modelling.csv')
print(df.head())
X=df.drop(labels=['CustomerId','Surname','RowNumber','Exited'],axis=1)
y=df['Exited']
lb=LabelEncoder()
X['Geography']=lb.fit_transform(X['Geography'])
X['Gender']=lb.fit_transform(X['Gender'])
X=pd.get_dummies(X,drop_first=True,columns=['Geography'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
reg=Sequential()
Reg=Sequential()
Reg.add(Dense(X.shape[1],activation='relu'))
reg.add(Dense(128,activation='relu'))
reg.add(Dense(1,activation='sigmoid'))
reg.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
reg.fit(X_train,y_train.to_numpy(),batch_size=10,epochs=2,verbose=1)
y_pred=reg.predict_classes(X_test)


print(y_pred,y_test)
reg.evaluate(X_test,y_test.to_numpy())
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))