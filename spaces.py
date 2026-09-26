import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from rescaling import rescaled_frame

img = cv.imread("D:\CV_practice\practice_images\dogs_4.jpg")
rescaled_img=rescaled_frame(img,0.5)
cv.imshow('BGR',rescaled_img) #This is by default a BGR image

#BGR to GRAY:
gray=cv.cvtColor(rescaled_img,cv.COLOR_BGR2GRAY)
#cv.imshow('GRAY',gray)

#BGR TO HSV(high Saturation Value):
hsv=cv.cvtColor(rescaled_img,cv.COLOR_BGR2HSV)
#cv.imshow('hsv',hsv)

#BGR to l*a*b(see notes):
lab=cv.cvtColor(rescaled_img,cv.COLOR_BGR2Lab)
#cv.imshow('lab',lab)

#BGR to RGB :
rgb=cv.cvtColor(rescaled_img,cv.COLOR_BGR2RGB)
cv.imshow('rgb',rgb)

#Showing difference in the image read by OpenCV and Matplot:
img_plt=plt.imread("D:\CV_practice\practice_images\dogs_4.jpg")

fig,axes=plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes=axes.flatten()
axes[0].imshow(img)#Here as we passed the image that is read by OpenCV it is converted to BGR format so for Matplot, 
# #the image(real) is an BGR image which is then converted as RGB by matplot itself  .
axes[0].set_title("image read by openCV displayed in Matplot")

axes[1].imshow(img_plt) #here as we passed the image that is read by matplot which is already in RGB format so 
#the image is displayed as it is 
axes[1].set_title("image read by Matplot")

plt.tight_layout()
plt.show()

cv.waitKey(0)