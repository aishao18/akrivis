import numpy as np
import cv2 as cv2

#The laplaciann
# matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Routine to fix 
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


image = cv2.imread("raw-chocolate.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(fixColor(image))

lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
plt.imshow(fixColor(lap))



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

# img = cv2.imread('raw-chocolate.jpg',0)
# edges = cv2.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()