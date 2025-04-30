import cv2;
import numpy as np;

'''
NOTE: ROTATION STEPS
'''
# image = cv2.imread("Dog and Cat.jpg")
image = cv2.imread("Dog and Cat.jpg", cv2.IMREAD_GRAYSCALE)

# dimension of the image
(h, w) = image.shape[:2] # no need for third channed

# centre of the image
center_x = w /2
center_y = h/ 2

center = (center_x, center_y)
# rotation angels in degree
angel = 90 
# the sacle (1.0 mean no scaling)
scale = 1.0   # 1.0 means don't make change on image˜

rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)  # FIRST; getting rotation matrix of the image
print(rotation_matrix)  

rotated_image = cv2.warpAffine(image, rotation_matrix, (w,h)) # apply any transformatino matrix on top of images 

'''
NOTE: NOISE
Noise types:
1. Gaussain* 2. Salt and Pepper* 3. Possion noise 4. Speckle noise 5. Thermal noise
'''

# Gaussain noise in CV2 (add noises to the image)
# using random from numpy (there are other random moduls in python as well but we use numpy)

if image is None:
              print("ERRO! image is not availabe")

gaussain_noise = np.random.normal(0, 25, image.shape).astype('float32')   # mean, standard diviation, size      


# cv2.imshow('frame', image + gaussain_noise)   it will work but it's not what we want from the image
noisy_image = cv2.add(image.astype('float32'),gaussain_noise)
noisy_image = np.clip(noisy_image,0,255).astype('uint8')  # 
cv2.imshow('frame', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Impulsive Salt and Pepper Noise in CV2
def add_noise(img: np.asarray) -> np.asarray:
              row, col, = img.shape
              
              numer_of_pixel = np.random.randint(300,200000) 
              # how much of my pixels in my image I want to change (lowerband,uppadband)
              # upper band cannot exceed the image size
              for i in range (numer_of_pixel):
                   y = np.random.randint(0, row-1)
                   x = np.random.randint(0,  col -1)
                   img[y,x] = 255   

              
              numer_of_pixel = np.random.randint(300,200000)
              for i in range (numer_of_pixel):
                   y = np.random.randint(0,row - 1)
                   x = np.random.randint(0,  col - 1)
                   
                   img[y, x] = 0  
              return img

noisy_image = add_noise(image)
cv2.imshow('frame', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

