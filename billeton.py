import cv2
import pathlib

route_img = str(pathlib.Path(__file__).parent.absolute() / 'images' ) 
print(route_img)

def load_img(name="billeton.png"):
    return cv2.imread(route_img+'/'+name)

def display_img(img,name="imagen"):
    cv2.imshow(name,img)
    cv2.waitKey()

img = load_img('billeton.png')
display_img(img)