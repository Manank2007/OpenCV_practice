import cv2 as cv
from rescaling import rescaled_frame
img =cv.imread("D:\CV_practice\practice_images\cats_1.jpg")
rescaled_image=rescaled_frame(img)
cv.imshow('image',rescaled_image)
gray=cv.cvtColor(rescaled_image,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray',gray)

#blurring an image:
blur=cv.GaussianBlur(rescaled_image,(7,7),cv.BORDER_DEFAULT) #the value of k_size is the amount of blur we want to apply , 
#so to increase the blur we have to increase the value of k_size but it can only be odd and positive integer as the function work by finding the center point of the grid and the even values do not have a proper center.
#cv.imshow('blur',blur)

#Edge Cascading :Basically displays the edges present in the images .
canny=cv.Canny(rescaled_image,125,250)
#cv.imshow('EDGE',canny)

#Dilating the image:Making the edges of the image sharper and adds pixels to boundaries of the objects .
dilated=cv.dilate(canny,(11,11),iterations=2)
#cv.imshow('dilated(11X11)',dilated)

#Eroding the image :toning down the sharpness of the edges of an image,makes it pretty close to the original edged image not completely restores the image :
eroded=cv.erode(dilated,(11,11),iterations=2)
#cv.imshow('eroded',eroded) 

#Resizing an image:
resized=cv.resize(img,(500,500),interpolation=cv.INTER_AREA) #here 500,500 is the desired resized image size 
cv.imshow('resized',resized)

#Cropping an image:
cropped=img[120:300,300:500] #it uses the fact that an image is an array so basically its just slicing of an array.
cv.imshow('cropped',cropped)
cv.waitKey(0)

