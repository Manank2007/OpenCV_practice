import cv2 as cv
import numpy as np
img=cv.imread( "D:\CV_practice\practice_images\cats_and_dogs.jpg") 
cv.imshow('img',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#Laplacian:  it is a method of edge detction that uses the second derivative of the image to find edges. It highlights regions of rapid intensity change and is often used to detect edges in images.
laplace=cv.Laplacian(gray,cv.CV_64F) #it gives a matrix of float 64 type including negative values too so 
laplace=np.uint8(np.absolute(laplace))# we have to convert it to uint8 format and also take the absolute value of the matrix


#Sobel 
sobelx=cv.Sobel(gray,cv.CV_64F,1,0) #shows the edges in the y direction (vertical edges)
sobely=cv.Sobel(gray,cv.CV_64F,0,1) #shows the edges in the x direction (horizontal edges)
combined_sobel=cv.bitwise_or(sobelx,sobely)

#Canny edge detection 
canny=cv.Canny(gray,150,205)
cv.imshow('canny',canny)
cv.imshow('sobelx',sobelx)
cv.imshow('sobely',sobely)
cv.imshow('combined_sobel',combined_sobel)
cv.imshow('Laplacian',laplace)
cv.waitKey(0)
