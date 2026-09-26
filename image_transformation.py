#Uncomment the waitkey and imshow when running . 
import cv2 as cv
import numpy as np
from rescaling import rescaled_frame

img=cv.imread("D:\CV_practice\practice_images\cats_4.jpg")

rescaled_image=rescaled_frame(img,0.5)
#cv.imshow('image',rescaled_image)

#Translating : it means to change the position of the image within the grid (moving it right or left and up or down).
def translate(img ,X ,Y):
    transMat=np.float32([[1,0,X],[0,1,Y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

#>> -x : shifting the image to left
#->> x : shifting the image to the right
#->> y : shifting the image downwards
#->> -y : shifting the image upwards
translated_image=translate(rescaled_image,100,100)
#cv.imshow('translated',translated_image)

#Rotating an image :
def rotate(img,angle,rotPoint=None):
    (height,width)=img.shape[:2]
    if rotPoint is None:
        rotPoint=(width//2,height//2)
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)
    return cv.warpAffine(img,rotMat,dimensions) 
#here Dsize is used to set the dimensions of the canvas so by using the 
#dimensions from the original image both of them will have same dimensions.
rotated=rotate(rescaled_image,45)
#cv.imshow('rotated',rotated)

#Flipping an image
flip=cv.flip(rescaled_image,1) #Can use 0 for vertical flip , 1 for horizoantal flip(mirror image), -1 for both vertical and horizontal.
#cv.imshow('flipped',flip)

#cv.waitKey(0)