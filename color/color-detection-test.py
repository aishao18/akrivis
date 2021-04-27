#Identify red objects in an image

#import OpenCV
import cv2
#Import numpy
import numpy as np

#open webcam
imgcap=cv2.VideoCapture(0)

while(1):

    #view the image from the webcam
    _, frame=imgcap.read()
    #convert the image to HSV
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #lower threshold for red
    lower_red=np.array([0, 100, 75])
    #upper threshold for red
    upper_red=np.array([5, 76, 100])

    mask=cv2.inRange(hsv, lower_red, upper_red)

    print("hey")
