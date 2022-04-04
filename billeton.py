# carpeta de drive con imagenes: 
# https://drive.google.com/drive/folders/1v4V4WuqLqjzT-V2j6jRMzxD7C8Bu8H2X

# biblioteca opencv para el procesamiento de imagenes

import cv2

# biblioteca para las rutas del sistema (integrada en python)

import pathlib

# biblioteca scikit-learn para los clasificadores

from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

# biblioteca matplotlib para graficas y otras utilidades matemáticas

import matplotlib.pyplot as plt

# biblioteca numpy para los arreglos y otras utilidades matemáticas

import numpy as np

# biblioteca pandas para los datasets 

import pandas as pd

# la ruta especifica de la imagen siendo "images" la ruta donde guardamos las imagenes
route_img = str(pathlib.Path(__file__).parent.absolute() / 'images' ) 
#print(route_img)

# billeton.png es el nombre de la imagen por defecto
# el metodo retorna el objeto cv2 de la imagen especificadoa
def load_img(name="billeton.png"):
    return cv2.imread(route_img+'/'+name,1)

# El metodo muestra la imagen especificada
# recibe un objeto cv2 y el nombre de la ventana, por defecto "imagen"
def display_img(img,name="imagen"):
    cv2.imshow(name,img)
    cv2.waitKey()

# INTER_NEAREST
# INTER_LINEAR
# INTER_AREA
# INTER_CUBIC
# INTER_LANCZOS4
# percent of original size
def resize_img(img,scale_percent = 200, i = cv2.INTER_AREA):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = i)
    return resized

def resize_img_res(img,width = 800,height = 600, i = cv2.INTER_AREA):
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = i)
    return resized

#img = load_img('billeton.png')
#display_img(img)

verdadero = load_img('Billetes_de_20/billete_20.jpg')
falso = load_img('Billetes_de_20/billete_F_20.png')
falso = resize_img(falso,400,cv2.INTER_LANCZOS4)

verdadero = resize_img_res(verdadero,1000,600,cv2.INTER_LANCZOS4)
falso = resize_img_res(falso,1000,600,cv2.INTER_LANCZOS4)

display_img(verdadero,"verdadero")
display_img(falso,"falso")

#print(type(verdadero))
#print(verdadero)
nverdadero = np.array(verdadero)
print(nverdadero)
df_v = pd.DataFrame(nverdadero.reshape(-1, 3), columns = ['red','green','blue'])
print(df_v)

"""
 Utility function to get the report of the accuracy of one model.
"""
def get_accuracy(model, X, y):
  cv = KFold(n_splits=10, random_state=1, shuffle=True)
  scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  print("Scores", scores)
  print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

nfalso = np.array(falso)
print(nfalso)
df_f = pd.DataFrame(nfalso.reshape(-1, 3), columns = ['red','green','blue'])
print(df_f)

exit()
#X_train, X_test, y_train, y_test = train_test_split(df_v.values, df_f.values, test_size=0.3, random_state=1) # 70% training and 30% test

classifier = linear_model.LinearRegression()
classifier.fit(df_v.values, df_f.values)
print("Score linear Regression: ", classifier.score(df_v.values, df_f.values))
#get_accuracy(classifier, df_v.values, df_f.values)

#classifier = DecisionTreeClassifier()
#classifier = classifier.fit(df_v.values, df_f.values)
#print("Decision tree classifier: ", classifier.score(df_v.values, df_f.values))

print("Logistic Regression")
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier.fit(df_v.values, df_f.values)
print("Score logistic Regression: ", classifier.score(df_v.values, df_f.values))