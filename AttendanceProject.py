import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Attendance_Images'   #importing all images
images = []   #creating list of all imported images
classNames = []    #to store names of all images
myList = os.listdir(path)    #list to store all names
print(myList)

# now from my list we will import images one by one and seperate name only from extension and store them one by one in images and their names in classnames

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)     #appending image to list
    classNames.append(os.path.splitext(cl)[0])  # Here we are splitting name from extension and storing only name in list
print(classNames)

def findEncodings(images):
    encodeList = [] # list to store encodings
    for img in images:   #now one by one we will loop through all images and change them from BGR to RGB and find their encodings
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #changing from BGR2RGB
        encode = face_recognition.face_encodings(img)[0]      #finding encodings
        encodeList.append(encode)  #appending encodings in encodings list
    return encodeList




def markAttendance(name):
    with open('Attendance.csv','r+') as f:     #  opening csv file in read and write mode
        myDataList = f.readlines()        # using this we can read all the lines of csv file that we opened
        nameList = []                 # we made a list so if the name is already in our csv file seperated by comma then nothing
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:     #if name is not in csv file then we extract current time and write name and current time seperated by comma in our csv file
            now = datetime.now()     #getting current time
            dtString = now.strftime('%H:%M:%S')    #format of time
            f.writelines(f'\n{name},{dtString}')   #writing name and time in file




encodeListKnown = findEncodings(images)   #In encode listknown we have encodings of all known faces
print('Encoding Completed')

cap = cv2.VideoCapture(0)     #starting webcam

while True:
    success, img = cap.read()    #this will extract image using webcam
    imgsmall = cv2.resize(img,(0,0),None,0.25,0.25)    #reducing size of image
    imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)   #again changing input image from bgr to rgb

    facesCurFrame = face_recognition.face_locations(imgsmall)    #locating faces
    encodesCurFrame = face_recognition.face_encodings(imgsmall,facesCurFrame)    #finding encoding of input image

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):  # to match the input with our data we will loop through it and match it with them
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)     #matching encodings of input and stored data
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)     #finding facedistance of input data in reference with stored data
        matchIndex = np.argmin(faceDis)   #it gives us index of stored image from the storage list

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()   # we are matching index and taking name of that image at that index and storing in name variable
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 =  y1*4,x2*4,y2*4,x1*4  #As we reduced size of our image we need to multiply it by 4 otherwise the rectangle will be drawn small
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)   #bounding box around the input
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)    #
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)    #writing the name extracted on the input image
            markAttendance(name)

        cv2.imshow('webcam',img)
        cv2.waitKey(1)




