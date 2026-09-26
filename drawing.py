import cv2 as cv
import numpy as np

blank=np.zeros((500,500,3),dtype='uint8')

# cv.imshow('blank',blank)      To display a blank canvas 
# blank[200:300,300:400]=0,0,255
# cv.imshow('red',blank)

# cv.rectangle(blank,(0,0),(250,250),(0,0,255),thickness=-1 )   >>>> To display a rectangle , thickness=-1 means filled shape
# cv.imshow('red_filled',blank)

cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2),60,(255,0,255),thickness=-1)
cv.imshow('circle',blank)

cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2),(0,0,255),thickness=3) #line doesnt take -1 as thickness
cv.imshow('line',blank)
#writing on an image
cv.putText(blank,'Hello everyone',(0,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow('text',blank)

cv.waitKey(0)