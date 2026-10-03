import cv2 as cv
import numpy as np
haar_cascade=cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# features = np.load('features.npy', allow_pickle=True) (No need as we are already importing the trained model which contains all the info about the labels and the features )

# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'D:\CV_practice\Faces\val\mindy_kaling\4.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
attempts = [                                      #stores a list of  multiple factors and neighbours 
    {'scaleFactor': 1.1, 'minNeighbors': 4},
    {'scaleFactor': 1.05, 'minNeighbors': 6},
    {'scaleFactor': 1.2, 'minNeighbors': 3},
]

best_label, best_confidence = None, float('inf')   #initialised best confidence to infinity so that the first iteration of confidence always gets stored in it 

for params in attempts:  #loops over each varaition in factor and neigbour size 
    faces_rect = haar_cascade.detectMultiScale(gray, **params)  #**params dtore the value of attempts at that particular index of params
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        if confidence < best_confidence:
            best_label, best_confidence = label, confidence

if best_confidence < 80:  #displays the names only if an confident match is found
    print(f'Match: {people[best_label]}, confidence={best_confidence}')
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow('Detected Face', img)
else:
 print('No confident match found')

cv.waitKey(0)