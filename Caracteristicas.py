import cv2
from cv2 import COLOR_BAYER_BG2RGB
from cv2 import COLOR_RGB2HSV
import numpy as np
import matplotlib.pyplot as plt

#Leemos la imágenes de los billetes
img=cv2.imread('images/Billetes_de_100/billete de 100_2.jpg')
imgf=cv2.imread('images/billete100fake.png')
imgf=cv2.resize(imgf,(965,527))


#Pre-análisis
'''
Usando la saturación podemos apreciar manera visual (por mientras, claro) las características que diferencian
un billete verdadero con uno falso
'''

img=cv2.cvtColor(img,COLOR_RGB2HSV)
imgf=cv2.cvtColor(imgf,COLOR_RGB2HSV)
cv2.imshow('Billeton',img[:,:,1])
cv2.imshow('Billeton Fake',imgf[:,:,1])
cv2.imshow('Billeton S',img[:,:,2])
cv2.imshow('Billeton Fake S',imgf[:,:,2])
cv2.imshow('Billeton V',img[:,:,0])
cv2.imshow('Billeton Fake V',imgf[:,:,0])
cv2.waitKey()

'''
Usar la función Threshold para crear la imágen binaria, lo único malo es que solo está escrito manualmente xd
'''
# recortereal = img[:,90:95,:]
# recortefalso = imgf[:,93:98,:]
# satThresh = 0.4
# valThresh = 0.3
# BWImageReal = ((recortereal[:,:,1] > satThresh).all() and (recortereal[:,:,2] < valThresh).all())
# # BWImageReal = plt.subplot(1,2,1) 
# cv2.imshow("real",BWImageReal) #<- Aquí truena
# BWImageFake = ((recortefalso[:,:,1] > satThresh).all() and (recortefalso[:,:,2] < valThresh).all())
# # BWImageFake = plt.subplot(1,2,2)
# cv2.imshow("False",BWImageFake)