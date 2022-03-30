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
    return cv2.imread(route_img+'/'+name)

# El metodo muestra la imagen especificada
# recibe un objeto cv2 y el nombre de la ventana, por defecto "imagen"
def display_img(img,name="imagen"):
    cv2.imshow(name,img)
    cv2.waitKey()

img = load_img('billeton.png')
display_img(img)