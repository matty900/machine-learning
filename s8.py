'''
NOTE DENOISING NOTE
Usage: improving clarity and quality of images
-----------------------------
Denoising Categories: 
              Techniques: Spatial : Mean, Median, Gaussian
              Frequency Domain Filtering
              Advance methods: Walvelet, NLM, Deep learning based denoising
Filter: Mean Filter, Gaussinan Filter, Median Filter, Bilateral Filter 
Kernel : small grid of numbers (e.g., a 3x3 or 5x5 matrix).
NOTE: The kernel :  WHAT feature to look for.,Convolution:  HOW the kernel is used to extract features.
Convolution: Convolution is the process of sliding the kernel across the image and performing element-wise multiplication and summation
Mean Denoise
-----------------------------
- FILTERING (convolution) (algorightm for denoising) + DEEP LEARNING
- Mean filtering
- convolution are type of filters BUT their mathmathical operation on counting the averga of pixels are differnt 
- convolution ( mathematical operation) come from singal communication
- NOISE: noise as random "specks" or distortions that degrade the quality of an image. 
It can be caused by things like low light conditions, sensor limitations, or compression artifacts.
- Imagine sliding a small window (called a "kernel" or "filter")
- Different kernels are used to detect different types of features (e.g., edges, corners, curves).
- convolutional neural network (CNN): demilshing image size(make it smaller)
-----------------------------
- How Kernel works : 
1. small matrix of random numbers 
2. Sliding over the image: The kernel "slides" over the input image, one pixel at a time.
3. Element-wise multiplication: At each position, the numbers in the kernel are multiplied with the corresponding pixel
values in the image
4. Summation: The results of the multiplication are added together to produce a single number
5. Feature map: This number is stored in a new grid called a "feature map" or "activation map."
5.1 CNN: Passing to the next layer: These feature maps become the input to the next layer in the CNN.
 The feature map highlights where the specific pattern that the kernel is designed to detect is present in the image
6. Error calculation: The network compares its output (the denoised image) to the actual clean image and calculates the error.
7. Backpropagation: The error is used to adjust the numbers in the kernels.
8. Iteration: The process repeats many times, with the network gradually 
refining the kernel values to become more and more effective at removing noises
-----------------------------
What the CNN Does with Feature Maps:
Multiple kernels, multiple maps: A CNN layer typically uses multiple kernels, each designed to detect a different feature. This results in multiple feature maps, each highlighting a different aspect of the image.   
Passing to the next layer: These feature maps become the input to the next layer in the CNN.
Deeper layers, complex features: Subsequent layers in the CNN learn to combine the features from the previous layers to detect even more complex patterns. For example, if the first layer detects edges, the second layer might learn to combine those edges to recognize shapes.   
Final layers and task: The final layers of the CNN are designed to perform the specific task the network is trained for (e.g., denoising, image classification). These layers might combine the high-level features from the earlier layers to make a decision.
'''
import cv2
import numpy as np

# image = cv2.imread("square.png") For detecting edges


'''make and apply noises '''

# noise = np.random.normal(0,25,image.shape).astype('float32')
# noisy_image = cv2.add(image.astype('float32'),noise)
# noisy_image = np.clip(noisy_image,0,255).astype('uint8')

'''Convolve Vertically (edge detection)'''
'''
The kernel is like a tiny magnifying glass that you use to find specific things in an image. 
Convolution is the act of sliding that magnifying glass across the image to find those things.
'''

# define kernel
# NOTE: only detect one edge. not all the edges together
#   The values [-1, 1] are specifically chosen to detect edges.
# kernel = np.array([[-1,1]]) # detec the left side edge # KERNEL 
# kernel = np.array([-1,1]) # detec the top side edge
# # apply filter (left to right, top to bottom ...)
# out_image = cv2.filter2D(image, cv2.CV_8U, kernel) # CONVOLUTION

# cv2.imshow("image", out_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Denoising
'''
# mean denoising 

image = cv2.imread("Pic.jpeg") # for denoising
noise = np.random.normal(0,25, image.shape).astype('float32')
noisy_image = cv2.add(image.astype('float32'), noise)
noisy_image = np.clip(noisy_image,0,255).astype('uint8')

kernel = np.ones((3*3)) * 1/9  # it can be also kernel = np.ones((3*3)) * 1/11 or anything else

filtered_image = cv2.filter2D(noisy_image, -1, kernel) # it reduced the noise

cv2.imshow("image", filtered_image)
# cv2.imshow("image", noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.blur() 
# cv2.rotate()