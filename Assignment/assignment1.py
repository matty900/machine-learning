import cv2
import numpy as np

# Question 2 
# Multiplying pixel values by a number less than 1 decreases the contrast
# Values closer to 0 make the image more gray and less contrasty

# a . Brightness and Contrast             

image = cv2.imread("Assignment/Dog and Cat.jpg") 
cv2.imshow("Original Image", image)
cv2.waitKey(0)

bright_image = cv2.add(image, np.ones(image.shape, dtype="uint8") * 150)
cv2.imshow("Brightened Image", bright_image)
cv2.waitKey(0)


contrast_image = cv2.multiply(image, np.array([0.5])) # [0.5] contrast factor.
cv2.imshow("Contrast Adjusted Image", contrast_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# b. Linear blend

# image2 = cv2.imread("S9_Lambo.png")  # Load second image
# image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
# cv2.imshow("Second Image", image2)
# cv2.waitKey(0)

# alpha = float(input("Enter alpha value (0 to 1): "))

# blend = cv2.addWeighted(image, 1 - alpha, image2, alpha, 0)
# cv2.imshow("Blended Image", blend)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


# Question 3 
# 1.2 : by chnaging thickness to -1  the rectangle will be fill out with color

# img = "Assignment/Dog and Cat.jpg"
# image = cv2.imread(img)
 
# cv2.rectangle(image,(384,0),(510,128),(0,255,0),-1)
# cv2.putText(image, "text", (420,80) , cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), thickness=2)
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.rectangle uses the top-left and bottom-right corners to define the rectangle,
# cv2.putText uses the bottom-left corner of the text as its reference point.