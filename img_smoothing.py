import cv2 as cv
import numpy as np 
img =cv.imread("D:\CV_practice\practice_images\cats_3.jpg")
cv.imshow('img',img)
#Averaging:it essentially takes the average of pixels in the kernel size window defined by 
# the user and apply that average to the middle pixel of the kernel winodw , then shifts the kernel by right and
# down so that whole image is covered 
average=cv.blur(img,(5,5)) #the bigger the window the more the blur (REMEMBER TO JUST TAKE ODD VALUES OF KERNEL SIZE)
cv.imshow('averaged',average)

#Gaussian Blur:See the notes for definiton .
gausssian=cv.GaussianBlur(img,(5,5),1)
cv.imshow('gaussian',gausssian)

#Median Blur:See notes.
median=cv.medianBlur(img,5)
cv.imshow('median',median)

#Bilateral Blurring:see the notes.
bilateral=cv.bilateralFilter(img,-1,15,15)
cv.imshow('bilateral',bilateral)

cv.waitKey(0)