import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
print(tf.__version__)
mnist = keras.datasets.fashion_mnist
(X_train, y_train),(X_test,y_test) = mnist.load_data()
#plt.imshow(X_train[0])
#plt.show()
X_train=X_train/255.0
X_test=X_test/255.0
reg=Sequential()
reg.add(Flatten(input_shape=(28,28)))
reg.add(Dense(128,activation='relu'))
reg.add(Dense(10,activation='softmax'))
print(reg.summary())
reg.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = reg.fit(X_train,y_train,epochs=10,batch_size=500,validation_split=.2)
test_loss,test_acc =reg.evaluate(X_test,y_test)
pred = reg.predict(X_test)
y_pred=reg.predict_classes(X_test)
print(accuracy_score(y_test,y_pred))
print(pred[0])
print(np.argmax(pred[0]))
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
print(class_labels[np.argmax(pred[0])])
#plt.imshow(X_test[0])
#plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_Accuracy')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
#### overfitting model as you can see the validation score is less than accuracy score###

#confusion matrix
mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=mat,class_names=class_labels)
plt.show()