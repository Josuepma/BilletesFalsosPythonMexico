from traceback import print_tb
import csv
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path
import os

def validacion(edge,edgec2):
    MIN_MATCH_COUNT = 8
    resultados=[]
    kp1, des1 = sift.detectAndCompute(edge,None)
    kp2, des2 = sift.detectAndCompute(edgec2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = edge.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(edgec2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    if(matchesMask!=None):
        #print(len(matchesMask))
        None
    else:
        matchesMask=[0,0]
    if(len(matchesMask)>5):
        resultados+=["1"]  
    else:
        resultados+=["0"]
      
    #print(draw_params)
    edgec2 = cv.drawMatches(edge,kp1,edgec2,kp2,good,None,**draw_params)
    #plt.imshow(edgec2, 'gray'),plt.show()
    return resultados

MIN_MATCH_COUNT = 8
resultados = []
resultadosc2 = []
resultadosc3 = []
resultadoFinal = []
billetes=[]


p = Path('images/Billetes_de_50')
for child in p.iterdir(): 
    #print(child)
    billetes.append(child.__str__())

p = Path('images/billeteF_50')
for child in p.iterdir():
    #print(child) 
    billetes.append(child.__str__())

#print(billetes)

for i in billetes:
    print(i)
    img1 = cv.imread(i,0)          # queryImage
    img4 = cv.imread('images/Caracteristica_50/C1.jpg',0) # trainImage
    img3 = cv.imread('images/Caracteristica_50/C2.jpg',0) # trainImage
    img5 = cv.imread('images/Caracteristica_50/CAR3.jpg',0) # trainImage
    edge=cv.Canny(img1,100,200)
    edgec2=cv.Canny(img3,100,200)
    edgec4=cv.Canny(img4,100,200)
    edgec5=cv.Canny(img5,100,200)
    sift = cv.SIFT_create()
    resultados+=validacion(edge,edgec2)
    resultadosc2+=validacion(edge,edgec4)
    resultadosc3+=validacion(edge,edgec5)
    if(int(resultados[-1]) == 1 and int(resultadosc2[-1]) == 1):
        resultadoFinal+=["1"]
    else:
        resultadoFinal+=["0"]

    # find the keypoints and descriptors with SIFT
    

print(resultados)
print(resultadosc2)
columnas = ['c1', 'c2', 'c3', 'b']

with open("Billetes50C1.csv", 'w', newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columnas)
    writer.writeheader()
    for i in range(len(resultados)):
        file.write(resultados[i] + ',' + resultadosc2[i] +
         ',' + resultadosc3[i] + ',' + resultadoFinal[i])
        file.write("\n")

def validacion(edge,edgec2):
    MIN_MATCH_COUNT = 8
    resultados=[]
    kp1, des1 = sift.detectAndCompute(edge,None)
    kp2, des2 = sift.detectAndCompute(edgec2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = edge.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(edgec2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    #print(len(good))
    if(matchesMask!=None):
        print(len(matchesMask))
    else:
        matchesMask=[0,0]

    if(len(good)>=2):
        resultados+=["1"]  
    else:
        resultados+=["0"]
      
    #print(draw_params)
    # edgec2 = cv.drawMatches(edge,kp1,edgec2,kp2,good,None,**draw_params)
    #plt.imshow(edgec2, 'gray'),plt.show()
    return resultados

resultados=[]
resultadosc2=[]
resultadosc3=[]
resultadoFinal = []
input_images_pathV = "images/Billetes_de_20"
files_namesV = os.listdir(input_images_pathV)
for file_nameV in files_namesV:
    print(file_nameV)    
    imgv = input_images_pathV + "/" + file_nameV
    img1 = cv.imread(imgv,0) #queryimage
    imgf=cv.resize(img1,(965,527))
    imgc1 = cv.imread('images/Caracteristica_20/caracteristica_1.jpg',0) # trainImage
    imgc2 = cv.imread('images/Caracteristica_20/caracteristica_2.jpg',0) # trainImage
    imgc3 = cv.imread('images/Caracteristica_20/caracteristica_3.jpg',0) # trainImage
    edge=cv.Canny(imgf,80,40)
    edgec1=cv.Canny(imgc1,80,40)
    edgec2=cv.Canny(imgc2,80,40)
    edgec3=cv.Canny(imgc3,80,40)
    sift = cv.SIFT_create()
    resultados+=validacion(edge,edgec1)
    resultadosc2+=validacion(edge,edgec2)
    resultadosc3+=validacion(edge,edgec3) 
    if(int(resultados[-1]) == 1 and int(resultadosc2[-1]) == 1):
        resultadoFinal+=["1"]
    else:
        resultadoFinal+=["0"] 

print(resultados, "\n", resultadosc2, "\n", resultadosc3, "\n", resultadoFinal)
print(len(resultados), len(resultadosc2), len(resultadosc3), len(resultadoFinal))
columnas = ['c1', 'c2', 'c3', 'b']

with open("Billetes20C.csv", 'w', newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columnas)
    writer.writeheader()
    for i in range(len(resultados)):
        file.write(resultados[i] + ',' + resultadosc2[i] +',' + resultadosc3[i] + ',' + resultadoFinal[i])
        file.write("\n")

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
