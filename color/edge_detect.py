#################################################
# import numpy as np
# import cv2 as cv2

# #The laplaciann
# # matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
#################################################

#Routine to fix 
# def fixColor(image):
#     cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# image = cv2.imread("raw-chocolate.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # plt.imshow(fixColor(image))

# lap = cv2.Laplacian(image, cv2.CV_64F)
# lap = np.uint8(np.absolute(lap))
# plt.imshow(fixColor(lap))



# sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# titles = ['Original Image', 'Combined',
#             'Sobel X', 'Sobel Y']
# images = [image, sobelCombined, sobelX, sobelY]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

######################################################
# img = cv2.imread('raw-chocolate.jpg',0)
# edges = cv2.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()
#######################################################



import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read the video
while(cap.isOpened()):
  # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
        # Converting the image to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('Frame', gray)

        # Converting the image to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using the Canny filter to get contours
        edges = cv2.Canny(gray, 20, 30)
        # Using the Canny filter with different parameters
        edges_high_thresh = cv2.Canny(gray, 60, 120)
        # Stacking the images to print them together
        # For comparison
        images = np.hstack((gray, edges, edges_high_thresh))

        # Display the resulting frame
        cv2.imshow('Frame', images)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()