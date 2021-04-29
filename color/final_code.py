import os
import cv2
import numpy as np
import time

def main():
    # Setup
    capWebcam = cv2.VideoCapture(0)

    capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)  # change resolution to 320x240 for faster processing
    capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

    lower = np.array([0, 0, 0], dtype = "uint8")
    upper = np.array([65, 54, 49], dtype = "uint8")

    if capWebcam.isOpened() == False:  # check if VideoCapture object was associated to webcam successfully
        print("error: capWebcam not accessed successfully\n\n")
        os.system("pause")
        return

    while(capWebcam.isOpened()):
        # Capture frame-by-frame
        ret, frame = capWebcam.read()
        if ret:
            # ret, frame = capWebcam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            all_pixels = cv2.inRange(frame, np.array([0, 0, 0], dtype = "uint8"), np.array([255, 255, 255], dtype = "uint8"))
            total_pixel_num = float(cv2.countNonZero(all_pixels))

            mask = cv2.inRange(hsv, lower, upper)

            # print("hry", total_pixel_num)

            output = cv2.bitwise_and(hsv, hsv, mask = mask)
            #cv2.imshow("images", np.hstack([hsv, output]))
            licorice = float(cv2.countNonZero(mask))

            print("dsfe ", licorice)
            print("pixel ratio hopefully", float(licorice/total_pixel_num))
            
            ratio = mask.size/hsv.size
            print(ratio)
            cv2.imshow("imagesssss", hsv)

            # images = np.hstack((mask, frame))

            # # Display the resulting frame
            # cv2.imshow('Frame', images)

    # When everything done, release the video capture object
    capWebcam.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()


main()