import cv2 as cv
#Reading images in openCV
#img=cv.imread("D:\CV_practice\practice_images\cats_1.jpg")
#cv.imshow('cat_1',img)
#cv.waitKey(0)


#Reading videos in openCV
capture=cv.VideoCapture("D:\CV_practice\practice_videos\dog_video1.mp4") #it can take integer arguments as well as path to a particular video, if we give it an integer path 
#it points to capturing video from a cam , either the laptops built in cam which ususally refred to as 0 or some external cam refred to as 1 or 2 etc .
while True:
    isTrue,frame=capture.read() #it stores the value from the capture.read() frame by frame inside a variable called 'frame'.
    cv.imshow('video',frame)

    if cv.waitKey(20) & 0XFF==ord('d'): #it means to stop the video if 'd' is pressed on the keeb.
        break
capture.release()
cv.destroyAllWindows() #the system will throw an error215 after running the whole video as there will be nomore frames toread.