# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:30:43 2017

@author: Akki
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('train.csv')

X = df.iloc[:, [2,4,5,9]]
y = df.iloc[:, 1].values
print(df["Survived"].value_counts(normalize = True))
print(df["Survived"][df["Sex"] == 'male'].value_counts(normalize = True))
X["Age"] = X["Age"].fillna(X.median()['Age'])

#X = X.fillna({"Embarked":'S'})
#train_two = X.copy()
X["family_size"] = df["SibSp"] + df["Parch"] + 1

X = pd.get_dummies(X,drop_first=True)

#split data

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Logistic regression
'''from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X,y)
y_pred = classifier.predict(X_test)'''
# Knn
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
'''
'''from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 19)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)'''
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

#decisiontree
'''
from sklearn import tree
my_tree_three = tree.DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
my_tree_three = my_tree_three.fit(X_train,y_train)
y_pred = my_tree_three.predict(X_test)
'''


# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth = 10, min_samples_split=2,n_estimators = 50,criterion = 'entropy',random_state = 0)
cl = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
p = (cm[0,0] + cm[1,1])/sum(sum(cm)) * 100

print(cl.feature_importances_)
print(cl.score(X_test,y_test))


# test set
df1 = pd.read_csv('test.csv')
X_test = df1.iloc[:, [1,3,4,8]]
X_test["Age"] = X_test["Age"].fillna(X_test.mean()['Age'])
X_test["Fare"] = X_test["Fare"].fillna(X_test.mean()['Fare'])
print(X_test[X_test.isnull().any(axis=1)])
X_test["family_size"] = df1["SibSp"] + df1["Parch"] + 1

X_test = pd.get_dummies(X_test,drop_first=True) 

y_pred = classifier.predict(X_test)

result = pd.DataFrame()
result['PassengerId'] = df1['PassengerId']

result['Survived'] = y_pred

result.to_csv('Titanic_RF.csv',index = False)
