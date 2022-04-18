from pickle import NONE
import cv2
from cv2 import imwrite
from cv2 import imread
import numpy as np
import os
import pathlib

def ajustar(imgr):
    imgf=cv2.resize(imgr,(1029,521))
    return imgf

def caracteristicas(imga):
    cv2.imshow("original", imga)
    #c1 = imga[50:300, 0:225]
    c2 = imga[100:500, 760:1029]
    #cv2.imwrite("C1.jpg", c1)
    car1=imread("C1.jpg",0)
    #cv2.imwrite("C2.jpg", c2)
    car2=imread("C2.jpg",0)
    c3 = imga[0:250, 560:740]
    cv2.imwrite("CAR3.jpg",c3)
    car3=imread("CAR3.jpg",0)
    cv2.imshow("C3",c3)
    orb=cv2.ORB_create(nfeatures=20)
    kp1,des1=orb.detectAndCompute(img,None)
    kp2,des2=orb.detectAndCompute(car3,None)
    kp3,des3=orb.detectAndCompute(car2,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches=bf.match(des1,des2)
    matches=sorted(matches,key=lambda x: x.distance)
    match_img=cv2.drawMatches(car3,kp2,img,kp1,matches[:50],None)
    tot_feature_matches = len(matches)
    print(f'Total Number of Features matches found are {tot_feature_matches}')
    cv2.imshow('CARACTERISTICA',match_img)
    cv2.waitKey()



img = cv2.imread('Billetes de 50/billete de 50.jpg',0)
imgfa = cv2.imread('billeteF_50/billeteF_50.jpg',0)
imgfa=ajustar(imgfa)
print(imgfa.shape) # Print image shape
#caracteristicas(img)

#ESTO NO ME FUNCIONA XDDDDD
route_img = str(pathlib.Path(__file__).parent.absolute() / 'images/Billetes de 50' ) 
print(route_img)
ruta="C:\Users\angel\Dropbox\PC\Desktop\Nuño\Unidad 3\10\images\Billetes de 50"
files_namesV = os.listdir(ruta)
for file_name in files_namesV:
    img2 = route_img + "/" + file_name
    print(img2)