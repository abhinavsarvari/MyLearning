import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense

print(tf.__version__)
mnist = keras.datasets.fashion_mnist
(X_train, y_train),(X_test,y_test) = mnist.load_data()
plt.imshow(X_train[0])
#plt.show()
X_train=X_train/255.0
X_test=X_test/255.0
reg=Sequential()
reg.add(Flatten(input_shape=(28,28)))
reg.add(Dense(128,activation='relu'))
reg.add(Dense(10,activation='softmax'))
print(reg.summary())
reg.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
reg.fit(X_train,y_train,epochs=1)
test_loss,test_acc =reg.evaluate(X_test,y_test)
pred = reg.predict(X_test)
print(pred[0])
plt.imshow(X_test[0])