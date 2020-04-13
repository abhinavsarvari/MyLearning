import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
digits=load_digits()
print(dir(digits))
plt.matshow(digits.images[69])
plt.show()
df=pd.DataFrame(digits.data)
df['target']=digits.target
X_train,X_test,y_train,y_test = train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)
#50 trees to be implemented and merged
reg=RandomForestClassifier(n_estimators=50)

reg.fit(X_train,y_train)
print(reg.predict([digits.data[69]]))
