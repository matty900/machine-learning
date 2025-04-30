import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical
from keras import models, layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import SGD

IMG_SIZE = (32, 32)

def preprocess_data():
    data_list = []
    label_list = []
    le = LabelEncoder()
    
    # Iterate through all images in the kapcha directories
    for i, address in enumerate(glob.glob('S23/kapcha/*/*.png')):
        img = cv2.imread(address)
        img = cv2.resize(img, IMG_SIZE)
        img = img/255.0  # Normalize pixel values to [0,1]
        data_list.append(img)
        # Extract label from directory name
        label = address.split('/')[-2]  # Gets the folder name (1-9)
        label_list.append(label)
        
        if i % 100 == 0:
            print(f'[INFO]: {i} images processed!')
    
    data_list = np.array(data_list)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2, random_state=42, stratify=label_list)
    
    # Encode labels to numerical values 
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # Convert labels to categorical (one-hot encoded) format
    num_classes = 9
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train, y_test

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocess_data()

# Data Augmentation
aug = ImageDataGenerator(
    rotation_range=20,       
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,      
    zoom_range=0.2,        
    horizontal_flip=True,  
    vertical_flip=True,   
    fill_mode="nearest",
    # brightness_range=[0.9, 1.1]  
)

# Enhanced CNN Model with Batch Normalization
net = models.Sequential([
    # First Conv 
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    # Second Conv 
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    # Third Conv 
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2)),
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(9, activation='softmax')  # 9 classes for digits 1 to 9
])

# Custom optimizer for learning rate
optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

# Compile the model
net.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train the model with data augmention
history = net.fit( aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), batch_size=64, epochs=20)

# Plot training and validation accuracy and loss
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("EVALUATION")
plt.show()

# Save the trained model
net.save('S23/kapcha_classifier.h5')
