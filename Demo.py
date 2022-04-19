import cv2
from traceback import print_tb
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
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    print(draw_params)
    if(matchesMask!=None):
        
        None
    else:
        matchesMask=[0,0]
    if(len(matchesMask)>=3):
        resultados+=["1"]  
    else:
        resultados+=["0"]

    edgec2 = cv.drawMatches(edge,kp1,edgec2,kp2,good,None,**draw_params)
    plt.imshow(edgec2, 'gray'),plt.show()
    return resultados
# cam = cv2.VideoCapture(0)

# cv2.namedWindow("test")

# img_counter = 0
# var = True
# while  var == True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         # img_counter += 1
#         var = None

# cam.release()
# cv2.destroyAllWindows()
resultados = []
resultadosc2 = []
resultadosc3 = []
resultadoFinal = []
img1 = cv.imread("opencv_frame_0.png",0)          # queryImage
# cv2.imshow("a ver", img1)
# cv2.waitKey()
img4 = cv.imread('images/Caracteristica_50/C1.jpg',0) # trainImage
img3 = cv.imread('images/Caracteristica_50/C2.jpg',0) # trainImage
img5 = cv.imread('images/Caracteristica_50/CAR3.jpg',0) # trainImage
edge=cv.Canny(img1,100,200)
edgec2=cv.Canny(img3,100,200)
edgec4=cv.Canny(img4,100,200)
edgec5=cv.Canny(img5,100,200)
cv2.imshow("a ver", edge)
cv2.waitKey()
sift = cv.SIFT_create()
resultados=validacion(edge,edgec2)
resultadosc2=validacion(edge,edgec4)
resultadosc3=validacion(edge,edgec5)
print(resultados, resultadosc2, resultadosc3)
if(int(resultados[-1]) == 1 and int(resultadosc2[-1]) == 1):
    resultadoFinal+=["1"]
    print("verdadero")
else:
    resultadoFinal+=["0"]
    print("Falso")