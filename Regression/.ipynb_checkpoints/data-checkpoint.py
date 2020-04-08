import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv('HousePrices.csv')


reg = linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)
reg.predict([[2000]])
print('The optimal line is '+ 'y = ' +  str(reg.coef_) + 'x + ' + str(reg.intercept_) )

plt.scatter(df.Area,df.Price)
plt.plot(df.Area,reg.predict(df[['Area']]),color='red')
plt.show()