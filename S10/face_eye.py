import cv2
import numpy as np

image = cv2.imread("s10/Pic.jpeg")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

face_cascade= cv2.CascadeClassifier('s10/haarcascades/haarcascade_frontalface_default.xml') 
eye_cascade= cv2.CascadeClassifier('s10/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
faces = face_cascade.detectMultiScale(imageGray, 1.1 , 4)

# NOTE: to be more accurate, we should first detect the face and then eyes

for (x, y, w, h) in faces: # draw a shape over the found face
              cv2.rectangle(image, (x, y), (x+w , y+h), (255,255,110),4)
              roi_image = imageGray[y:y+h, x:x+w]
              eyes = eye_cascade.detectMultiScale(roi_image, 1.1 , 4)

for (x2, y2 ,w2, h2) in eyes:
        cv2.circle(image, (x+x2+w2//2, y+y2+h2//2), 10 , (0,255,0,4))
# print(faces)

cv2.imshow("me",image)
cv2.waitKey(0)
cv2.destroyAllWindows()