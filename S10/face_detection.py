import cv2
import numpy as np

image = cv2.imread("s10/Pic.jpeg")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # should turn these to gray for next step

face_cascade= cv2.CascadeClassifier('s10/haarcascades/haarcascade_frontalface_default.xml') # use the model(algorithm)

faces = face_cascade.detectMultiScale(imageGray, 1.1 , 4) # detect the face in the image

for (x, y, w, h) in faces: # draw a shape over the found face
              cv2.rectangle(image, (x, y), (x+w , y+h), (255,255,110),4)

# print(faces)

cv2.imshow("me",image)
cv2.waitKey(0)
cv2.destroyAllWindows()