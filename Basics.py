import cv2
import numpy as np
import face_recognition

imgPrince = face_recognition.load_image_file('Basic_Images/Prince Raj.JPG')
imgPrince = cv2.cvtColor(imgPrince,cv2.COLOR_BGR2RGB)
# The upper code is to bring normal image in and encode over it

imgTest = face_recognition.load_image_file('Basic_Images/Prince_Test.JPG')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
# The upper image is to test encodings done by above code and see if we can recognize test image

faceLoc = face_recognition.face_locations(imgPrince)[0] #it will return us 4 values that are coordinates of face detected
encodePrince = face_recognition.face_encodings(imgPrince)[0]
cv2.rectangle(imgPrince,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0] #it will return us 4 values that are coordinates of face detected
encodePrinceTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodePrince],encodePrinceTest)
#This upper results code uses encoding of both real and test images and matches them and returns true if they match and false if they don't.

faceDis = face_recognition.face_distance([encodePrince],encodePrinceTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
#Facedis is the distance between reference image and test image which means lower the distance more the accuracy


cv2.imshow('Prince Raj',imgPrince)
cv2.imshow('Prince Test',imgTest)

cv2.waitKey(0)
