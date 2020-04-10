import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import joblib

df = pd.read_csv('HousePrices.csv')


reg = linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)
print(reg.predict([[2000]]))
print('The optimal line is '+ 'y = ' +  str(reg.coef_) + 'x + ' + str(reg.intercept_) )

with open ('model_pickle','wb') as f:
    pickle.dump(reg,f)
    
with open ('model_pickle','rb') as f:
    md= pickle.load(f)

print(md.predict([[5000]]))

joblib.dump(reg,'model_joblib')
jb=joblib.load('model_joblib')
print(jb.predict([[7777]]))

