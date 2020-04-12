import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
digits=load_digits()
print(dir(digits))
print(digits.data[0])
plt.gray()
#plt.matshow(digits.images[0])
#plt.show()
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
reg=LogisticRegression()
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))
#reg.predict()
#plt.matshow(digits.images[100])
#plt.show()
print(reg.predict([digits.data[100]]))
#code for checking for the wrong results
y_predicted=reg.predict(X_test)
Conf_matrix=confusion_matrix(y_test,y_predicted)
print(Conf_matrix)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(Conf_matrix,annot=True)
plt.show()