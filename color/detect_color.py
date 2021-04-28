# import the necessary packages
import numpy as np
import argparse
import cv2
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
	#([0, 0, 0], [0, 0, 0])
	#([65, 65, 65], [65, 65, 65])
	([0, 0, 0], [200, 200, 200])
]


# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	num_whites = np.sum(image <= 200)
	num_blacks = np.sum(image == 0)

	print('Number of white pixels: ',num_whites)
	print('Number of black pixels: ',num_blacks)
	print("Lower: ", lower)
	print("Upper: ", upper)
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.imshow("images", mask)
	cv2.waitKey(0)
