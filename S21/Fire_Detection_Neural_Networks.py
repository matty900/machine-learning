import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.src.utils import to_categorical # instead of keras.utils used this option 
from keras import models, layers
'''
Encoding : Converting each label to a number. for instnce cat -> 0 , dog -> 1, bird -> 3 . sckit learn do this automatically , 
However in keras (TensorFlow) we need to do that menually .
Type of encoding: One Hot Encoding . Instead of  cat -> 0 , dog -> 1, bird -> 3 that make priority steps  we use Cat [1 0 0 0 ] , dog [0 1 0 0 ] bird [0 0 1 0]
This will eliminate the priority step . using : from keras.utils import to_categorical
STEPS: 1. transfer to label by LabelEncounter 2. use keras  to_categorical  
------------
Ways to defining Neural Network:
1. Sequential 
2. Fanctional API (Advance)
3. Model Subclassing (Advance)
------------
in Neural Network everything is relative. nothing is absolute. It means for instance if something is 5.1 would be map to 0.90 probabilities. which means it 90% can be classified
We take the maximum probabilites for clasification. 
Step for NN:
Design
Define 
Fit
Compile 
'''
IMG_SIZE = (32, 32)

# DATA (in Porject)
def preprocess_data():
    data_list = []
    label_list = []
    le = LabelEncoder()

    for i, address in enumerate(glob.glob('S18/fire_dataset\\*\\*')):
        img = cv2.imread(address)
        img = cv2.resize(img, IMG_SIZE)
        img = img/255 
        img = img.flatten()

        data_list.append(img)

        label = address.split('\\')[-1].split('.')[0]
        label_list.append(label)

        if i%100 == 0:
            print(f'[INFO]: {i}/1000 processed!')

    data_list = np.array(data_list)

    X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2)

    y_train = le.fit_transform(y_train) # for train we use fit_tranform, which for normalization
    y_test = le.transform(y_test)  # for test we transform()
    
    # One Hot Encoding (for neural netword is needed)
    y_train = to_categorical(y_train) 
    y_test = to_categorical(y_test)

    print(y_train)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data()

# NOTE: The most important algorithm in AI, Machine learning.... is  -> SGD -> (for optimaziton) (The best optimzed algorithm)

# MODEL
# importing sequential way to defining neural network
net = models.Sequential([
                        layers.Dense(20, activation='sigmoid'), # (efining neurons, sigmoid for derivitave calculation )
                        layers.Dense(8, activation='sigmoid'),
                        layers.Dense(2, activation='softmax') # softmax instead of sigmoid
                        ])


net.compile(optimizer='SGD', # optimzation function
            loss='categorical_crossentropy', # loss function for classification for instance: MSE, MAE , or Cross entropy Loss
            metrics='accuracy') # accuracy score 

H = net.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10) # batch_size -> how many time check the data to update one label # epochs: group size
# by having batch_size=64, epochs=10 it means updated 64 times in 10 size of the data 


net.save('S21/net.h5')

