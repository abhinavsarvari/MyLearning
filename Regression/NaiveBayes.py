import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
target=df['Survived']
inputs=df.drop('Survived',axis='columns')
dummies=pd.get_dummies(inputs.Sex)
inputs=pd.concat([inputs,dummies],axis='columns')
print(inputs.head(3))
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
inputs.drop(['Sex','male'],axis='columns',inplace=True)
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3)
model=GaussianNB()
model.fit(X_train,y_train)
print(model.predict([[1,50,5,0]]))
print(model.score(X_test,y_test))
print(model.predict_proba(X_test[:10]))
#calculate score using cross validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(GaussianNB(),X_train,y_train,cv=5))