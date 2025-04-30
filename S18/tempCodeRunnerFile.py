knn = KNeighborsClassifier(n_neighbors=5)  # Try tuning this parameter
# knn.fit(X_train, y_train)
# knn_predictions = knn.predict(X_test)

# knn_accuracy = accuracy_score(y_test, knn_predictions)
# print(f'KNN Accuracy: {knn_accuracy:.4f}')
# print(classification_report(y_test, knn_predictions))

# # Save KNN model
# dump(knn, 'cat_dog_knn_model.z')