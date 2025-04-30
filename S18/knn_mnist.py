import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
train = pd.read_csv('S18/MNIST_Dataset/mnist_train.csv')
test = pd.read_csv('S18/MNIST_Dataset/mnist_test.csv')

# NOTE: No need the operation such as flattening like on images

# Split feature and labels
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)

# Save the model
joblib.dump(knn, 'knn_mnist_model.pkl')
print("Model saved as 'knn_mnist_model.pkl'")

# Load the model 
loaded_model = joblib.load('knn_mnist_model.pkl')


y_pred = loaded_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy: {accuracy * 100:.2f}%')

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
