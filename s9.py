import cv2;
import numpy as np;

'''
NOTE FEATURE EXTRACRION NOTE
Features are distinct patterns in images 
Usage: in identifying objects, textures, and structures (for object detection, recognize region of interest)
-----------------------------
Types:
Color Features: RGB, HSV, or other color space information
Shape: Shapes and format
Edges: Borders between different regions
Corners: Points with high curvature
Blobs: Regions with uniform texture or intensity
Keypoints: Important points used for matching
Textures: Patterns formed by pixel intensities
-----------------------------
Methods:
CLASSICAL METHODS: SIFT, SURF, ORB, HOG
DEEP LEARNING BASED FEATURES: CNN FEATURE MAPS
-----------------------------
Algorithms:
Thresholding => range to detect specific colors
Histogram Analysis => uses the color histogram of an image to detectdominant colors
Gaussian Mixture Models => probabilistic approach that models pixel distributions for different colors
ML based (K-Means clustering) => groups similar colors into clusters and highlight the dominant color
DL based => a model can be trained to classify and segment colors

The Thresholding is not that effective. so we want more automatic methods.
A > COLOR EXTRACTION 
'''
# A > COLOR EXTRACTION 

# def empty(val):
#         pass            

# cv2.namedWindow("TrackBar")
# cv2.resizeWindow("TrackBar", 640, 250)
# # use for color segmetation: Hue , saturation, value
# cv2.createTrackbar("Hue Min", "TrackBar", 0 , 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBar", 0 , 255, empty)
# cv2.createTrackbar("Val Min", "TrackBar", 0 , 255, empty)
# cv2.createTrackbar("Hue Max", "TrackBar", 0 , 179, empty)
# cv2.createTrackbar("Sat Max", "TrackBar", 0 , 255, empty)
# cv2.createTrackbar("Val Max", "TrackBar", 0 , 255, empty)

# while True: # 1. for reading the frames
#         image = cv2.imread("S9_Lambo.png")
        
#         # 2. change the color to hsv image
#         hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
       
#         # 3. get the values for track bars
#         h_min = cv2.getTrackbarPos("Hue Min", "TrackBar")  
#         s_min = cv2.getTrackbarPos("Sat Min", "TrackBar")
#         v_min = cv2.getTrackbarPos("Val Min", "TrackBar")
#         h_max = cv2.getTrackbarPos("Hue Max", "TrackBar")  
#         s_max = cv2.getTrackbarPos("Sat Max", "TrackBar")
#         v_max = cv2.getTrackbarPos("Val Max", "TrackBar")
        
#         # 4. define upper band and lower band to get the mask
#         # always ask user for lower and upper for trashhoulding values
#         low = np.array([h_min,s_min,v_min]) 
#         up = np.array([h_max,s_max,v_max]) 
        
#         # 5. everything in this range will be filtered. Basically creating mask
#         mask = cv2.inRange(hsv_image, low, up) 
 
#         # 6. bitwise the orginal image
#         image_result = cv2.bitwise_and(image, image, mask = mask)
       
#         cv2.imshow('image frame', image)
#         cv2.imshow('Mask' , mask)
#         cv2.imshow('hsv image', hsv_image)
#         cv2.imshow('result image', image_result)
#         cv2.waitKey(1)





# cv2.imshow('TrackBar', image)
# cv2.imshow('image_frame', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def empty(a):
    pass

path = 'S9_lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Original",img)  
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    cv2.waitKey(1)

