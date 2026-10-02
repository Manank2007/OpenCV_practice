import cv2 as cv
import numpy as np
img=cv.imread( "D:\CV_practice\practice_images\cats_and_dogs.jpg") 
cv.imshow('cats and dogs',img)
blank=np.zeros((img.shape[:2]),dtype='uint8')
cv.imshow('blank',blank)
#masking
circle=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
cv.imshow('circle',circle)
masked=cv.bitwise_and(img,img,mask=circle)

cv.imshow('masked',masked)
cv.waitKey(0)