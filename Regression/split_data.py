import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv('CarData.csv')
import matplotlib.pyplot as plt
plt.scatter(df['Mileage'],df['Sell Price'])
X=df[['Mileage','Age']]
y=df['Sell Price']
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
reg = LinearRegression()
reg.fit(Xtrain,ytrain)
result = reg.predict(Xtest)
print (Xtest)
print (result)
print(reg.score(Xtest,ytest))