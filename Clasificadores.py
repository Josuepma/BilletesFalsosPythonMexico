# carpeta de drive con imagenes: 
# https://drive.google.com/drive/folders/1v4V4WuqLqjzT-V2j6jRMzxD7C8Bu8H2X

# biblioteca opencv para el procesamiento de imagenes

import cv2

# biblioteca para las rutas del sistema (integrada en python)

import pathlib

# biblioteca scikit-learn para los clasificadores

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# biblioteca matplotlib para graficas y otras utilidades matemáticas

import matplotlib.pyplot as plt

# biblioteca numpy para los arreglos y otras utilidades matemáticas

import numpy as np

# biblioteca pandas para los datasets 

import pandas as pd


def get_accuracy(model, X, y):
  cv = KFold(n_splits=10, random_state=1, shuffle=True)
  scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  print("Scores", scores)
  print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


df = pd.read_csv('Billetes50C1.csv')
target = df['b']
del df['b']
print("|| Billetes de 50||")
X_train, X_test, y_train, y_test = train_test_split(df.values, target, test_size=0.3, random_state=27)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("\nDecision tree classifier: ", classifier.score(X_train, y_train))
get_accuracy(classifier, df.values, target)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("\nRandomForest classifier")
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train,y_train)
get_accuracy(classifier, df.values, target)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

print("\nmlp classifier")
classifier = MLPClassifier(max_iter=400)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train, y_train)
get_accuracy(classifier, X_train, y_train)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("KNeighborsClassifier")
model = KNeighborsClassifier(3)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("\nAda")
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train, y_train)
get_accuracy(classifier, X_train, y_train)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

# Initialize classifier:
gnb = GaussianNB()

# Train the classifier:
model = gnb.fit(X_train, y_train)
# Make predictions with the classifier:
predictive_labels = gnb.predict(X_test)
print(predictive_labels)
y_pred = model.predict(X_test)
# Evaluate label (subsets) accuracy:
print(accuracy_score(y_test, predictive_labels))
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


df = pd.read_csv('Billetes20C.csv')
target = df['b']
del df['b']
print("|| Billetes de 20||")
X_train, X_test, y_train, y_test = train_test_split(df.values, target, test_size=0.3, random_state=27)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("\nDecision tree classifier: ", classifier.score(X_train, y_train))
get_accuracy(classifier, df.values, target)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("\nRandomForest classifier")
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train,y_train)
get_accuracy(classifier, df.values, target)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

print("\nmlp classifier")
classifier = MLPClassifier(max_iter=400)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train, y_train)
get_accuracy(classifier, X_train, y_train)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("KNeighborsClassifier")
model = KNeighborsClassifier(3)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))


print("\nAda")
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train, y_train)
get_accuracy(classifier, X_train, y_train)
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

# Initialize classifier:
gnb = GaussianNB()

# Train the classifier:
model = gnb.fit(X_train, y_train)
# Make predictions with the classifier:
predictive_labels = gnb.predict(X_test)
print(predictive_labels)
y_pred = model.predict(X_test)
# Evaluate label (subsets) accuracy:
print(accuracy_score(y_test, predictive_labels))
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
