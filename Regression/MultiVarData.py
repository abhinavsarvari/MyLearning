'''This code is for multiple variables m1x1+m2x2+m3x3+b=0'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv('MultiVarHousingPrices.csv')
df.Bedrooms.fillna(df.Bedrooms.median(),inplace=True)
reg = linear_model.LinearRegression()
reg.fit(df[['Area','Bedrooms','Age']],df.Price)

print('The optimal line is '+ 'y = ' +  str(reg.coef_) + 'x + ' + str(reg.intercept_) )

print(reg.predict([[2800,5,1]]))

m1=reg.coef_[0]
m2=reg.coef_[1]
m3=reg.coef_[2]
x1=df.Area
x2=df.Bedrooms
x3=df.Age
plt.plot(df.Price,m1*x1+m2*x2+m3*x3+reg.intercept_)
plt.show()
'''plt.plot(df.Area,reg.predict(df[['Area']]),color='red')'''