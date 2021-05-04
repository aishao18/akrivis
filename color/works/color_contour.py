import cv2
import numpy as np
import argparse


frameWidth = 640
frameHeight = 480

# For figures
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# For videos
# cap = cv2.VideoCapture(0)

# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    contours, hierarchy  = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

while True:
    # success, img = cap.read()
    # img = cv2.imread(args["image"])
    # imgContour = img.copy()
    # imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    # imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    # imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    # kernel = np.ones((5, 5))
    # imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    # getContours(imgDil,imgContour)
    # imgStack = stackImages(0.3,([img,imgCanny],
    #                             [imgDil,imgContour]))


    lower = [80, 140, 180]
    # upper = [143, 203, 239]
    upper = [133, 193, 229]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # image = cv2.imread('figures/raw-chocolate.jpg')
    capWebcam = cv2.VideoCapture('figures/convey.mp4')
    ret, image = capWebcam.read()
    print("heyyyyyy ", ret)
    if ret:
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        maskContour = output.copy()


        imgBlur = cv2.GaussianBlur(image, (7, 7), 1)
        # imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgBlur,threshold1,threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)


        # maskBlur = cv2.GaussianBlur(mask, (7, 7), 1)
        # maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        maskCanny = cv2.Canny(mask,threshold1,threshold2)
        kernel = np.ones((5, 5))
        maskDil = cv2.dilate(maskCanny, kernel, iterations=1)
        getContours(maskDil, maskContour)
        # getContours(mask, maskContour) #maybe works
        # getContours(maskContour, imgDil)
        maskStack = stackImages(0.3,([mask,maskCanny],
                                    [maskDil,maskContour]))
        # output = cv2.bitwise_and(image, image, mask = mask)
        # # show the images
        # cv2.imshow("images", np.hstack([mask, maskCanny]))
        # cv2.waitKey(0)


        cv2.imshow("Result", maskStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("boooo")
        break



 

 
