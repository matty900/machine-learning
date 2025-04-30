import cv2

cap = cv2.VideoCapture("s10/S10_cars_video.mp4")
plate_cascade = cv2.CascadeClassifier('s10/haarcascades/haarcascade_russian_plate_number.xml')


while True:
              success, img = cap.read()
              img = cv2.resize(img, (1000,500))
              img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              plates = plate_cascade.detectMultiScale(img_gray , 1.1 , 4)

              for (x,y, w, h) in plates:
                      if (w*h > 10000) and (w*h < 50000): # for detecting the plates by sharing size
                            cv2.rectangle(img , (x,y), (x+w, y+h), (122, 9 ,32), 3)

              cv2.imshow('result', img)
              cv2.waitKey(1)
