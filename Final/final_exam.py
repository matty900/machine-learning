import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset

df = pd.read_csv('Final/dataset.csv')  

print("Statistical Summary")
print(df.describe())

print("\nClass Distribution (Target 'y')")
print(df['y'].value_counts(normalize=True))

print("\nCorrelation Matrix (Numeric Features)")
print(df.corr(numeric_only=True))


# Preprocessing

X = df.drop('y', axis=1)
y = df['y']

# Normalizing

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model Training

# model = RandomForestClassifier(max_iter=1000)
# model.fit(X_train, y_train)  # Random Forest doing better by accuary around 90%
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction & Evaluation

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))
