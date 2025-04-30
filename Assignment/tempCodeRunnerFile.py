image = cv2.imread("Assignment/Dog and Cat.jpg") 
cv2.imshow("Original Image", image)
cv2.waitKey(0)

bright_image = cv2.add(image, np.ones(image.shape, dtype="uint8") * 150)
cv2.imshow("Brightened Image", bright_image)
cv2.waitKey(0)

contrast_image = cv2.multiply(image, np.array([0.5]))
cv2.imshow("Contrast Adjusted Image", contrast_image)
cv2.waitKey(0)
cv2.destroyAllWindows()