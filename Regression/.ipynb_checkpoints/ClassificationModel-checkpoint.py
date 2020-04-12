import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('InsuranceData.csv')
plt.scatter(df.age,df.bought_insurance,marker='*',color='blue')
X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=.1)
reg=LogisticRegression()
reg.fit(X_train,y_train)
print(X_test)
print(y_test)
print(reg.predict(X_test))
print(reg.score(X_test,y_test))