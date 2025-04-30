import cv2
import numpy as np
from joblib import load

# Load model
model = load('cat_dog_knn_model.z')  # or 'cat_dog_log_reg_model.z'

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten().reshape(1, -1)
    
    prediction = model.predict(img)[0]
    label = 'Cat' if prediction == 0 else 'Dog'
    print(f'Prediction: {label}')

# Test 
predict_image('S18/Cat_Dog_Dataset/test/Cat/Cat (1).jpg')
predict_image('S18/Cat_Dog_Dataset/test/Dog/Dog (1).jpg')
