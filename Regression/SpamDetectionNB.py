import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
df=pd.read_csv('Spam.csv')
#print(df.groupby('Category').describe())
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
X_train,X_test,y_train,y_test=train_test_split(df.Message,df.spam)
v=CountVectorizer()
X_train_count=v.fit_transform(X_train.values)
#to check how data looks like
#df1=pd.DataFrame(X_train_count.toarray(),columns=v.get_feature_names())
reg=MultinomialNB()
reg.fit(X_train_count,y_train)
emails=[ 'Hey There, How are you doing?', 'we have 20% discount on our flats'] 
emails_count = v.transform(emails)
#print(reg.predict(emails_count))
pipe=Pipeline([('vectorizer',CountVectorizer()),('nb',MultinomialNB())])
pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))
print(pipe.predict(emails))