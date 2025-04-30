import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

def load_data():
    data_list = []
    labels = []
    
    # Load training data
    for label in ['Cat', 'Dog']:
        for address in glob.glob(f'S18/Cat_Dog_Dataset/train/{label}/*.jpg'):
            img = cv2.imread(address)
            img = cv2.resize(img, (64, 64)) 
            img = img / 255.0  # Normalize pixel values
            img = img.flatten()
            
            data_list.append(img)
            labels.append(0 if label == 'Cat' else 1)  # Cat = 0, Dog = 1

    data_list = np.array(data_list)
    labels = np.array(labels)

    return train_test_split(data_list, labels, test_size=0.2, random_state=42)

# Load data
X_train, X_test, y_train, y_test = load_data()

# KNN 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f'KNN Accuracy: {knn_accuracy:.4f}')
print(classification_report(y_test, knn_predictions))

# Save KNN model
dump(knn, 'cat_dog_knn_model.z')

# Logistic Regression 
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)

log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
print(f'Logistic Regression Accuracy: {log_reg_accuracy:.4f}')
print(classification_report(y_test, log_reg_predictions))

# Save LR model
dump(log_reg, 'cat_dog_log_reg_model.z')
