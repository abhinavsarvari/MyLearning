import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('town_prices.csv')
dummies=pd.get_dummies(df.town)
merged=pd.concat([df,dummies],axis='columns')
final = merged.drop(['town','west windsor'],axis='columns')
reg = LinearRegression()
X=final.drop('price',axis='columns')
print(X)
y=final.price
reg.fit(X,y)
print(reg.predict([[2800,0,1]]))
'''for west windsor we need to provide 0 0 '''
print('The accuracy of the data is ' + str(reg.score(X,y)*100)+'%')

'''CODE FOR ONE HOT ENCODER'''

