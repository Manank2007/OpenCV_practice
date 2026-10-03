import os 
import cv2  as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR= r'D:\CV_practice\Faces\train' #just the path to the faces directory

haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)#specifies a full path to the directory of that person by automatically including a (\ or / )
        label=people.index(person) 
    for img in os.listdir(path):  #loop over all the images present in the folder of a particular name 
        img_path=os.path.join(path,img) #to point to a specific image in the folder of each person

        img_array=cv.imread(img_path)
        if img_array is None:
                continue 
        gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

        faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces_rect:
              faces_roi = gray[y:y+h, x:x+w] #gives the specific face area only 
              features.append(faces_roi)
              labels.append(label)

create_train()
print("Training done -----------------")

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#training the recogniser on the labels and features

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

