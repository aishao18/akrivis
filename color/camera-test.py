# OpenCV_test_3.py

# this program tracks a red ball
# (no motor control is performed to move the camera, we will get to that later in the tutorial)

import os

import cv2
import numpy as np


###################################################################################################
def main():
    capWebcam = cv2.VideoCapture(0)  # declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

    # show original resolution
    print("default resolution = " + str(capWebcam.get(cv2.CAP_PROP_FRAME_WIDTH)) + "x" + str(
        capWebcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)  # change resolution to 320x240 for faster processing
    capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

    # show updated resolution
    print("updated resolution = " + str(capWebcam.get(cv2.CAP_PROP_FRAME_WIDTH)) + "x" + str(
        capWebcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if capWebcam.isOpened() == False:  # check if VideoCapture object was associated to webcam successfully
        print("error: capWebcam not accessed successfully\n\n")  # if not, print error message to std out
        os.system("pause")  # pause until user presses a key so user can see error message
        return  # and exit function (which exits program)
    # end if
    while cv2.waitKey(1) != 27 or capWebcam.isOpened():  # until the Esc key is pressed or webcam connection is lost
        blnFrameReadSuccessfully, imgOriginal = capWebcam.read()  # read next frame

        if not blnFrameReadSuccessfully or imgOriginal is None:  # if frame was not read successfully
            print("error: frame not read from webcam\n")  # print error message to std out
            os.system("pause")  # pause until user presses a key so user can see error message
            break  # exit while loop (which exits program)
        # end if

        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

        imgThreshLow = cv2.inRange(imgHSV, np.array([0, 135, 135]), np.array([18, 255, 255]))
        imgThreshHigh = cv2.inRange(imgHSV, np.array([130, 135, 135]), np.array([179, 255, 255]))
	sharpen_kernel = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])
	sharpen = cv2.filter2D(imgHSV,-1, sharpen_kernel)

        imgThresh = cv2.add(imgThreshLow, imgThreshHigh)

        imgThresh = cv2.GaussianBlur(imgThresh, (3, 3), 2)

        imgThresh = cv2.dilate(imgThresh, np.ones((5, 5), np.uint8))
        imgThresh = cv2.erode(imgThresh, np.ones((5, 5), np.uint8))

        intRows, intColumns = imgThresh.shape

        circles = cv2.HoughCircles(imgThresh, cv2.HOUGH_GRADIENT, 5,
                                   intRows / 4)  # fill variable circles with all circles in the processed image

        if circles is not None:  # this line is necessary to keep program from crashing on next line if no circles were found
            for circle in circles[0]:  # for each circle
                x, y, radius = circle  # break out x, y, and radius
                print("ball position x = " + str(x) + ", y = " + str(y) + ", radius = " + str(
                    radius))  # print ball position and radius
                cv2.circle(imgOriginal, (x, y), 3, (0, 255, 0),
                           -1)  # draw small green circle at center of detected object
                cv2.circle(imgOriginal, (x, y), radius, (0, 0, 255), 3)  # draw red circle around the detected object
                # end for
        # end if
	boundaries = [
		([17, 15, 100], [50, 56, 200]),
		([86, 31, 4], [220, 88, 50]),
		([25, 146, 190], [62, 174, 250]),
		([103, 86, 65], [145, 133, 128])
	]
	lower = np.array([103, 86, 65], dtype = "uint8")
	upper = np.array([145, 133, 128], dtype = "uint8")

	#https://stackoverflow.com/questions/60212888/printing-the-percentage-of-a-color-in-an-image-in-opencv-python

	################################################################
	# HERE IS THE GOOD STUFF TO TRY TO DETECT BLACK PIXELS
	ret, frame = capWebcam.read()
	hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower, upper)
	output = cv2.bitwise_and(hsv, hsv, mask = mask)
	cv2.imshow("images", np.hstack([hsv, output]))
	#################################################################

	# for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		#lower = np.array(lower, dtype = "uint8")
		#upper = np.array(upper, dtype = "uint8")
		#print("Lower: ", lower)
		#print("Upper: ", upper)
		# find the colors within the specified boundaries and apply
		# the mask
		#mask = cv2.inRange(hsv, lower, upper)
		#output = cv2.bitwise_and(hsv, hsv, mask = mask)
		# show the images
		#cv2.imshow("images", np.hstack([hsv, output]))
		#cv2.waitKey(0)

        cv2.namedWindow("imgOriginal", cv2.WINDOW_NORMAL)  # create windows, use WINDOW_AUTOSIZE for a fixed window size
        # cv2.namedWindow("imgThresh", cv2.WINDOW_NORMAL)  # or use WINDOW_NORMAL to allow window resizing
        # cv2.namedWindow("sharpen", cv2.WINDOW_NORMAL)  # or use WINDOW_NORMAL to allow window

        cv2.imshow("imgOriginal", imgOriginal)  # show windows
        # cv2.imshow("imgThresh", imgThresh)
	# cv2.imshow('sharpen',sharpen)

    # end while

    cv2.destroyAllWindows()  # remove windows from memory

    return



main()
