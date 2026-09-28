import cv2 as cv
import numpy as np
video=cv.VideoCapture('D:\CV_practice\practice_videos\dog_video1.mp4')
while True:
    ret,frame=video.read()
    if not ret:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur=cv.blur(frame,(5,5),0)
    cv.imshow('original',frame)
    cv.imshow('blur',blur)
    cv.imshow('gray',gray)

    if cv.waitKey(20) & 0XFF== ord('d'):
        break

video.release()
cv.destroyAllWindows()