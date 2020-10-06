import cv2
import numpy as np
import face_recognition
import os


path = 'attendance'
images = []
classnames = []
imglist = os.listdir(path)
print(imglist)
for itm in imglist:
    curImg = cv2.imread(f'{path}/{itm}')
    images.append(curImg)
    classnames.append(os.path.splitext(itm)[0])

print(classnames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encodelist)
    return(encodelist)

encodelistofknownfaces = findEncodings(images)
print('encoding successful')

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    imgSmall = cv2.resize(img, (0,0),None, 0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceincurframe = face_recognition.face_locations(imgSmall)
    encodecurframe = face_recognition.face_encodings(imgSmall,faceincurframe)

    for encodeface,faceloc in zip(encodecurframe,faceincurframe):
        matches = face_recognition.compare_faces(encodelistofknownfaces,encodeface)
        faceDis = face_recognition.face_distance(encodelistofknownfaces,encodeface)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches(matchIndex):
            name = classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2.y2),(0,255,0),cv2.FILLED )
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
