# import the necessary packages
import numpy as np
import argparse
import cv2
#from matplotlib import pyplot as plt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
# load the image
image = cv2.imread(args["image"])

# define the list of boundaries
boundaries = [
	#([17, 15, 100], [50, 56, 200])
	#([86, 31, 4], [220, 88, 50]),
	#([25, 146, 190], [62, 174, 250]),
	#([103, 86, 65], [145, 133, 128]),
	# ([0, 0, 0], [30, 30, 30])
	#([0, 0, 0], [0, 0, 0])
	#([65, 65, 65], [65, 65, 65])
	#([0, 0, 0], [200, 200, 200]),
	([0, 0, 0], [65, 54, 49])
	#([0, 0, 0], [0, 0, 0])
	# ([0, 0, 0], [255, 255, 255])
	#([0, 0, 0], [40, 40, 40])
]


# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	num_whites = np.sum(image <= 200)
	num_blacks = np.sum(image == 0)

	print('Number of total pixels: ',num_whites)
	print('Number of black pixels: ',num_blacks)
	print("Lower: ", lower)
	print("Upper: ", upper)

	dst = cv2.inRange(image, lower,upper)
	no_red = cv2.countNonZero(dst)
	print('The number of black pixels: ',no_red)

	all_pixels = cv2.inRange(image, np.array([0, 0, 0], dtype = "uint8"), np.array([255, 255, 255], dtype = "uint8"))
	total_pixel_num = float(cv2.countNonZero(all_pixels))
	
	print('The ratio: ', no_red/total_pixel_num)

	# test = image.total()
	# print("TEEEESSST: ", test)
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images

	the_ratio = "The black pixel ratio: " + str(round((no_red/total_pixel_num)*100,2)) + "%"

	cv2.putText(image, the_ratio, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
	cv2.imshow("Original",image)
	cv2.imshow("images", np.hstack([image, output]))
	#print('The hstack:')
	#print(np.hstack([image, output]))
	cv2.imshow("images", mask)
	cv2.waitKey(0)
