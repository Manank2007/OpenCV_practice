#Splitting the image into its color channels i.e. Blue, Green and Red
#the whiter areas in the greyscaled images represent more pixel intensity at that place of 
#that particular color channel

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rescaling import rescaled_frame
img=cv.imread("D:\CV_practice\practice_images\dogs_4.jpg")
img_scaled=rescaled_frame(img,0.5)
cv.imshow('img',img_scaled)

blank=np.zeros(img_scaled.shape[:2],dtype='uint8')
b,g,r=cv.split(img_scaled)

cv.imshow('b_grey',b) #This converts the image into a single color channel image which will look like grey image  
cv.imshow('g_grey',g)
cv.imshow('r_grey',r)

blue=cv.merge([ b,blank,blank]) #it merges the splitted image(which was broken into b,g,r ) in 3 channel format so 
#that it does not look like grey image (grey image is a single color channel image).Here 'blank' is used as a filler for
# the other 2 color channels
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])
original_img=cv.merge([b,g,r])
cv.imshow('orign_img',original_img)

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

cv.waitKey(0)