from multiprocessing.connection import wait
from traceback import print_tb
import csv
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
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
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    print(len(good))
    if(matchesMask!=None):
        print(len(matchesMask))
    else:
        matchesMask=[0,0]

    if(len(good)>=2):
        resultados+=["1"]  
    else:
        resultados+=["0"]
      
    print(draw_params)
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
    imgv = input_images_pathV + "/" + file_nameV
    img1 = cv.imread(imgv,0) #queryimage
    imgf=cv.resize(img1,(965,527))
    imgc1 = cv.imread('caracteristica_1.jpg',0) # trainImage
    imgc2 = cv.imread('caracteristica_2.jpg',0) # trainImage
    imgc3 = cv.imread('caracteristica_3.jpg',0) # trainImage
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
        file.write(resultados[i] + ',' + resultadosc2[i] +
         ',' + resultadosc3[i] + ',' + resultadoFinal[i])
        file.write("\n")
exit()