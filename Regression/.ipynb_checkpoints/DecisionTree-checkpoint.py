import pandas as pd
from sklearn import tree
df=pd.read_csv('Salaries.csv')
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
#print(inputs_n)
reg=tree.DecisionTreeClassifier()
reg.fit(inputs_n,target)
reg.score(inputs_n,target)
print(reg.predict([[2,2,1]]))
