import cv2 as cv
import numpy as np

blank=np.zeros((400,400),dtype='uint8')
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle=cv.circle(blank.copy(),(200,200),200,255,-1)
cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)

#OR operator:includes both intersection as well as non intersection areas
img_or=cv.bitwise_or(rectangle,circle)
cv.imshow("OR",img_or)

#AND operator:intersection points:
img_AND=cv.bitwise_and(rectangle,circle)
cv.imshow('AND',img_AND)

#XOR operator : give only the non intersection points of the images
img_XOR=cv.bitwise_xor(circle,rectangle)
cv.imshow('XOR',img_XOR)

#NOT operator : inverts the color of an image (takes only one image )
img_NOT=cv.bitwise_not(circle)
cv.imshow('NOT',img_NOT)

cv.waitKey(0)