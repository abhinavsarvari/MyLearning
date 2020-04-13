from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
digits=load_digits()
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.3)
#lr=LogisticRegression()
#lr.fit(X_train,y_train)
#print(lr.score(X_test,y_test))
svm=SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
kf=KFold(n_splits=3)
for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)
def get_score(reg,X_train,X_test,y_train,y_test):
    reg.fit(X_train,y_train)
    return reg.score(X_test,y_test)

scores_logistic = []
scores_svm = []
scores_rf = []
folds=StratifiedKFold(n_splits=3)
for train_index,test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

print(scores_logistic,scores_svm,scores_rf)

####Easy Method CrossVal function#####
print(cross_val_score(SVC(),digits.data,digits.target))