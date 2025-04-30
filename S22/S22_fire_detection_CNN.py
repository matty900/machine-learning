import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical
from keras import models, layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator

IMG_SIZE = (32, 32)

# DATA
def preprocess_data():
    """
    Loads images from the specified directory, resizes them, normalizes pixel values,
    encodes labels, and splits the data into training and testing sets.
    """
    data_list = []
    label_list = []
    le = LabelEncoder()
    # Iterate through all images in the specified directories
    for i, address in enumerate(glob.glob('S18/fire_dataset\\*\\*')):
        img = cv2.imread(address)
        img = cv2.resize(img, IMG_SIZE)
        img = img/255 
        # img = img.flatten()

        data_list.append(img)

        label = address.split('\\')[-1].split('.')[0]
        label_list.append(label)

        if i%100 == 0:
            print(f'[INFO]: {i}/1000 processed!')

    data_list = np.array(data_list)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2)
    
    # Encode labels to numerical values
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # Convert labels to categorical (one-hot encoded) format
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_train)
    return X_train, X_test, y_train, y_test

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocess_data()

# DATA AUGMENTATION
aug = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")
# CNN MODEL DEFINITION
net = models.Sequential([
                        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
                        layers.BatchNormalization(), # adds the normalization layer
                        layers.MaxPool2D(),
                        layers.Conv2D(32, (3,3), activation='relu'),
                        layers.BatchNormalization(),
                        layers.MaxPool2D(),
                        layers.Flatten(),
                        layers.Dense(100, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(2, activation='softmax')
                        ])

# Learning rate could be optimizer = SGD(learning_rate=0.001)  
# net.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
# COMPILE THE MODEL
net.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics='accuracy')

    # H = net.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test)) Instease of this we use

# applies the augmentation.
# TRAIN THE MODEL WITH DATA AUGMENTATION
H = net.fit(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), batch_size=64, epochs=10)

# PLOT TRAINING AND VALIDATION ACCURACY AND LOSS
plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="test")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("EVALUTATION")

# SAVE THE TRAINED MODEL
net.save('S22/CNN_classifier.h5')

