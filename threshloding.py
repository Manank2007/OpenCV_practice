import cv2 as cv
import numpy as np
from rescaling import rescaled_frame

img =cv.imread("D:\CV_practice\practice_images\dogs_2.jpg")
rescaled_img=rescaled_frame(img,0.5)

gray=cv.cvtColor(rescaled_img,cv.COLOR_BGR2GRAY)


#thresholding : Basically binariaing the image  pixels (0 or 255) based on a thereshold value set by the user.

#Simple thresholding:
threshold, thresh= cv.threshold(gray,150, 255,cv.THRESH_BINARY)

#Inverse thresholding:changes the pixel values less than the threshold value to 255 and the values in between to 0;
threshold_inverse,inverse_thresh=cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)


#Adaptive thresholding: basically it calculates the threshold value without us specifying the minimum threshold value :
adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,1)
#here 11 is the block size ( THE SIZE OVER WHICH THE MEAN OF THE PIXELS ARE CALCULATED) and 1 is the contsant that is subtracted from the mean as a fine tuning offset.



cv.imshow('inverse threshold',inverse_thresh)

cv.imshow('adaptive threshold',adaptive_thresh)
cv.imshow('THRESH_BINARY', thresh)
cv.imshow('gray', gray)
cv.imshow('img', rescaled_img)

cv.waitKey(0)
