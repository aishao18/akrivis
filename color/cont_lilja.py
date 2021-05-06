import cv2
import numpy as np
import time

frameWidth = 640
frameHeight = 480
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('figures/convey.mp4')
cap.set(3, frameWidth)
cap.set(4, frameHeight)
img_counter = 0


def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
cv2.createTrackbar("LowerArea","Parameters",160000,300000,empty)
cv2.createTrackbar("UpperArea","Parameters",280000,300000,empty)

cv2.createTrackbar("LowerX","Parameters",0,250,empty)
cv2.createTrackbar("UpperX","Parameters",90,250,empty)
cv2.createTrackbar("LowerY","Parameters",0,250,empty)
cv2.createTrackbar("UpperY","Parameters",30,250,empty)

cv2.createTrackbar("LicoriceThreshold","Parameters", 5, 1000, empty)

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

def getContours(img,imgContour,img_counter):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("LowerArea", "Parameters")
        areaMax = cv2.getTrackbarPos("UpperArea", "Parameters")
        if area > areaMin and area < areaMax:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            
            # print(len(approx))

            # x y bullid
            x , y , w, h = cv2.boundingRect(approx)
            if len(approx) >= 4:
                print("x ", x)
                print("y ", y)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

            lx = cv2.getTrackbarPos("LowerX","Parameters")
            ux = cv2.getTrackbarPos("UpperX","Parameters")
            ly = cv2.getTrackbarPos("LowerY","Parameters")
            uy = cv2.getTrackbarPos("UpperY","Parameters")

            compare = cv2.getTrackbarPos("LicoriceThreshold","Parameters")

            if len(approx) >= 4 and lx <= x and x <= ux and ly <= y and y <= uy:
                # print('it detects')
                _, image = cap.read()

                # TODO: check if image taken

                # TODO: analyze image and save if defective
                # img_name = "opencv_frame_{}.png".format(img_counter)
                # cv2.imwrite(img_name, imgContour)
                # print("{} written!".format(img_name))

                # image = cv2.imread("opencv_frame_{}.png".format(img_counter))
                # img_counter += 1


                lower = [0, 0, 0]
                upper = [65, 54, 49]

                # lower = np.array([0, 0, 0], dtype = "uint8")
                # upper = np.array([65, 54, 49], dtype = "uint8")


                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                # find the colors within the specified boundaries and apply
                # the mask
                mask = cv2.inRange(image, lower, upper)
                output = cv2.bitwise_and(image, image, mask = mask)

                all_pixels = cv2.inRange(image, np.array([0, 0, 0], dtype = "uint8"), np.array([255, 255, 255], dtype = "uint8"))
                total_pixel_num = float(cv2.countNonZero(all_pixels))
                licorice = float(cv2.countNonZero(mask))

                print("Pixel ratio: ", float(licorice/total_pixel_num))

                ratio = round((licorice/total_pixel_num)*10000,2)

                the_ratio_str = "The black pixel ratio: " + str(round((licorice/total_pixel_num)*10000,2)) + "%"
                print(the_ratio_str)

                # Pixel detection check
                if ratio >= compare:
                    print("Alert!!!!")
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, imgContour)
                    print("{} written!".format(img_name))
                    print(img_counter)
                    img_counter = img_counter + 1
                    # time.sleep(1) DOESNT WORK, VIDEO LAG

                # show the images
                cv2.imshow("images", np.hstack([image, output]))
                # cv2.waitKey(0)
    return img_counter

def main():
    img_counter = 0
    while True:
        success, img = cap.read()
        imgContour = img.copy()
        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        img_counter = getContours(imgDil,imgContour,img_counter)
        imgStack = stackImages(0.8,([img,imgCanny],
                                    [imgDil,imgContour]))
        cv2.imshow("Result", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()