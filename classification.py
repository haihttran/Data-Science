import sys
import pandas as pd
import matplotlib
import scipy as sp
import IPython
import sklearn
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
import os
from sklearn.model_selection import train_test_split
path = os.path.dirname(os.path.abspath(__file__))

#Disable pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

#Data loading, preprocessing, and cleaning step
dataset = pd.read_csv("train.csv")
#The training data using only 6 columns of the original dataset, remove unneccessary ones
train_data = dataset[['national_inv','lead_time','forecast_3_month','sales_6_month','min_bank','perf_6_month_avg']]
#Dealing with missing value in 'lead_time' column, replace NA with numeric 0
train_data['lead_time'] = train_data['lead_time'].fillna(0)
#Prepare train target
train_target = dataset[['went_on_backorder']]
train_target['went_on_backorder'].replace(('Yes', 'No'), (1, -1), inplace=True)

# Prepare test data; load, preprocess, and clean data
test = pd.read_csv("D:/Data/test.csv")
test_data = test[['national_inv','lead_time','forecast_3_month','sales_6_month','min_bank','perf_6_month_avg']]
test_data['lead_time'] = test_data['lead_time'].fillna(0)
#Remove header row
test_data.drop(test_data.head(0).index, inplace=True)
test_target = pd.read_csv("D:/Data/result.csv", header=None)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size = 0.1)
# print(test_data)
# print(test_target)

#First Model: Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

predict = clf.predict(test_data)
pd.DataFrame(predict).to_csv("test_set_sample_predictions.csv", sep="\n", index=False, header=False)
print("Cross-validation accuracy of Decision Tree: ", accuracy_score(y_test, clf.predict(X_test), "\n"))
#print(predict)
print("Test accuracy of Decision Tree: ", accuracy_score(test_target, predict), "\n")

#Second Model: KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.values.ravel())

predict2 = clf.predict(test_data)
print("Cross-validation accuracy of KNN: ", accuracy_score(y_test, knn.predict(X_test), "\n"))
#print(predict2)
print("Test accuracy of KNN: ", accuracy_score(test_target, predict2), "\n")

#Third Model: SVM
svm_clf = svm.SVC()
svm_clf.fit(X_train, y_train.values.ravel())

predict3 = svm_clf.predict(test_data)
print("Cross-validation accuracy of SVM: ", accuracy_score(y_test, svm_clf.predict(X_test), "\n"))
#print(predict3)
print("Test accuracy of SVM: ", accuracy_score(test_target, predict3), "\n")

#Fourth Model: Naive Bayes
lr = LogisticRegression()
lr.fit(X_train, y_train.values.ravel())

predict4 = lr.predict(test_data)
print("Cross-validation accuracy of Logistic Regression: ", accuracy_score(y_test, lr.predict(X_test), "\n"))
#print(predict4)
print("Test accuracy of Logistic Regression: ", accuracy_score(test_target, predict4), "\n")
