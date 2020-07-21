import sys
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

feature_vectors_df = pd.read_csv('labeled-feature-vectors.csv', sep=',', header=0)

X = feature_vectors_df.drop(columns=['class','buggy'], axis=1)
y = feature_vectors_df.buggy


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2) 

''' Default version '''
clf = DecisionTreeClassifier()
gnb = GaussianNB()
lsvc = LinearSVC()
mlpc = MLPClassifier()
rfc = RandomForestClassifier()
biased = DummyClassifier(strategy='constant', constant=1)

''' Fine-tuned version '''
# clf = DecisionTreeClassifier(criterion='entropy', splitter='random', presort=True)
# gnb = GaussianNB(var_smoothing=1e-3)
# lsvc = LinearSVC(loss='hinge', random_state=1)
# mlpc = MLPClassifier(hidden_layer_sizes=1000, activation='tanh', solver='sgd', learning_rate='adaptive')
# rfc = RandomForestClassifier(criterion='entropy', oob_score=True, warm_start=True)
# biased = DummyClassifier(strategy='constant', constant=1)


y_pred_clf = clf.fit(X_train, y_train).predict(X_test)
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
y_pred_lsvc = lsvc.fit(X_train, y_train).predict(X_test)
y_pred_mlpc = mlpc.fit(X_train, y_train).predict(X_test)
y_pred_rfc = rfc.fit(X_train, y_train).predict(X_test)
y_pred_biased = biased.fit(X_train,y_train).predict(X_test)


prfs_clf = precision_recall_fscore_support(y_test, y_pred_clf, average='binary')
prfs_gnb = precision_recall_fscore_support(y_test, y_pred_gnb, average='binary')
prfs_lsvc = precision_recall_fscore_support(y_test, y_pred_lsvc, average='binary')
prfs_mlpc = precision_recall_fscore_support(y_test, y_pred_mlpc, average='binary')
prfs_rfc = precision_recall_fscore_support(y_test, y_pred_rfc, average='binary')
prfs_biased = precision_recall_fscore_support(y_test, y_pred_biased, average='binary')

print()
print()
print("Decision Tree: ", prfs_clf)
print("Naive Bayes: ", prfs_gnb)
print("Support Vector Machine: ", prfs_lsvc)
print("Multi-Layer Perceptron: ", prfs_mlpc)
print("Random Forest: ", prfs_rfc)
print("Biased: ", prfs_biased)
print()
print()

'''Default results'''
# Decision Tree:  (0.4666666666666667, 0.5, 0.4827586206896552, None)
# Naive Bayes:  (0.4, 0.14285714285714285, 0.21052631578947364, None)
# Support Vector Machine:  (0.37142857142857144, 0.9285714285714286, 0.5306122448979592, None)
# Multi-Layer Perceptron:  (0.5, 0.21428571428571427, 0.3, None)
# Random Forest:  (0.4, 0.14285714285714285, 0.21052631578947364, None)
# Biased:  (0.25, 1.0, 0.4, None)

'''Fine-tuned results'''
# Decision Tree:  (0.2631578947368421, 0.35714285714285715, 0.30303030303030304, None)
# Naive Bayes:  (0.3333333333333333, 0.07142857142857142, 0.11764705882352941, None)
# Support Vector Machine:  (0.45454545454545453, 0.7142857142857143, 0.5555555555555556, None)
# Multi-Layer Perceptron:  (1.0, 0.07142857142857142, 0.13333333333333333, None)
# Random Forest:  (0.25, 0.07142857142857142, 0.11111111111111112, None)
# Biased:  (0.25, 1.0, 0.4, None)

