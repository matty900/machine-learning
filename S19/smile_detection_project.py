import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import SGDClassfier
from sklearn.metrics import accuracy_score, classification_report
from mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')

# we want to detect the face and classify them by ml algorithm into possitive or nagative faces

# hardcascade and MTCNN for face detection 

#  Machine learning algorithm
# decision tree, SVM/SVDD, random forest, PCA . logistic regression, classification, (gradient descent: optimizaiton)

IMAGE_SIZE = (32,32)

detector = MTCNN()
# Functions to detect to face in images to minimze the noise (anything else than face consider to be noise)
def face_detector(img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mtccn read rgb but cv2 is bgr so conversion is needed
        out = detector.detect_faces(img)[0]
        x, y, w, h = out['box']
        face = img[y:y+h, x:x+w ]
        return face
#DATA
def data_preprocess():
              data = []
              labels = []
              for i, address in enumerate(glob.glob('S19/smiile_dataset/*/*')):
                      print(address)
                      img = cv2.imread(address)
                      face = face_detector(img)
                      img = cv2.resize(img, IMAGE_SIZE)
                      img = img / 255
                      img = img.flatten()

                      data.append(img)

                      label = address.split('/')[1].split('\\')[1]
                      labels.append(label)
                      # Keep track of processed data
                      if i %100 == 0:
                            print(f'[INFO] {i}/3300 has ben processed')
              data = np.array(data)
              X_train, X_text, y_train, y_test = train_test_split(data,labels,shuffle=True, test_size=5)

              return X_train, X_text, y_train, y_test 


X_train, X_text, y_train, y_test = data_preprocess()


# # MODEL
clf = SGDClassfier()
clf.fit(X_train,y_train)
data_preprocess()



# EVALUATE

predecitions = clf.predict(X_text)
print(accuracy_score(X_text))





