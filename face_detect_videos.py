import cv2 as cv
import numpy as np
from rescaling import rescaled_frame
vid=cv.VideoCapture("D:\\CV_practice\\practice_videos\\7882906-uhd_3840_2160_30fps.mp4")
frame_counts=[]
while True:
    ret,frame=vid.read()
    if not ret:
        break
    haar_Cascade=cv.CascadeClassifier("haar_face.xml")
    frame_resized=rescaled_frame(frame,0.3)
    gray=cv.cvtColor(frame_resized,cv.COLOR_BGR2GRAY)
    face_rect=haar_Cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)
    frame_counts.append(len(face_rect))
    for (x,y,w,b) in face_rect:
       cv.rectangle(frame_resized,(x,y),(x+w,y+b),(0,0,255),thickness=2)
    
    cv.imshow('face detcting vid',frame_resized)

    if cv.waitKey(20) & 0XFF== ord('d'):
     break
print(f'Min faces in a frame: {min(frame_counts)}')
print(f'Max faces in a frame: {max(frame_counts)}')
print(f'Average faces per frame: {sum(frame_counts)/len(frame_counts):.2f}')
vid.release()
cv.destroyAllWindows()



