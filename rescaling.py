import cv2 as cv
def rescaled_frame(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
#The example code :
    capture=cv.VideoCapture("D:\CV_practice\practice_videos\dog_video1.mp4")
# while True:
#     isTrue,frame=capture.read()
#     frame_resized=rescaled_frame(frame)
#     cv.imshow('video',frame)
#     cv.imshow('video_resized',frame_resized)
#     if cv.waitKey(20) & 0XFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows


