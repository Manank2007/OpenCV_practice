import cv2 as cv
import pandas as pd
import numpy as np

img=cv.imread("D:\CV_practice\practice_images\cats_3.jpg")
blank=np.zeros(img.shape,dtype='uint8')
cv.imshow('blank',blank)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('gray',gray)

blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)

canny=cv.Canny(img,125,200)
cv.imshow('edges',canny)

ret,thresh=cv.threshold(blur,125,240,cv.THRESH_BINARY) #Thresholding is basically converting the pixels to binary format i.e., 
#below thresh1=(black),above thresh2=(white)
#cv.imshow('thresholding',thresh)

contours,heirarchies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print("contours = ",len(contours))
blank_contours=cv.drawContours(blank,contours,-1,(255,0,0),1)
cv.imshow('blank_contours',blank_contours)

cv.waitKey(0)