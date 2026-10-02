import cv2 as cv
import numpy as np
from rescaling import rescaled_frame
img=cv.imread("D:\\CV_practice\\practice_images\\group_photos.jpg") 
rescaled_img=rescaled_frame(img,0.65)
gray =cv.cvtColor(rescaled_img,cv.COLOR_BGR2GRAY)

cv.imshow('gray',gray)
#Face detection by haar cascading :
haar_Cascade=cv.CascadeClassifier("haar_face.xml")

faces_rect=haar_Cascade.detectMultiScale(gray,scaleFactor=1.07,minNeighbors=4)

for (x,y,w,h) in faces_rect:

 cv.rectangle(rescaled_img,(x,y),(x+w,y+h),(0,0,255), thickness=2)  #THE PARAMETERS OF FACES FROM GRAY SCALED IMGES ARE USED TO DRAW RECTANGLE IN THE NORMAL IMAGE.
print("the number of faces are :",len(faces_rect))
cv.imshow('detected faces',rescaled_img)
cv.waitKey(0)