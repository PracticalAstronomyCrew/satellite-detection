# Olivier Paredes
# Practical Astronomy Crew
####################################################

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from skimage.morphology import remove_small_objects
from astropy.io import fits
from skimage.transform import hough_line, hough_line_peaks

####################################################

# Open the FITS files
for subdir, dir, files in os.walk('../data/faint_images/'):
	for filename in files:
		if '.FIT' in filename:

			####################################################
			# Open the FITS files

			hdul = fits.open(f"../data/faint_images/{filename}")

			# Get the data and header information
			data = hdul[0].data
			header = hdul[0].header

			zscale = ZScaleInterval(contrast=0.1)

			vmin, vmax = zscale.get_limits(data)

			fig, ax = plt.subplots(figsize=(data.shape[1]/200, data.shape[0]/200))
			plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

			im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='gray')
			plt.axis('off')
			# redraw the canvas
			fig.canvas.draw()
			# convert canvas to image using numpy
			img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
			img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			# opencv format
			img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			img = ~img

			####################################################
			# Thresholding

			# for faint images chanage the last parameter to lowest value
			threshGaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 25)

			####################################################
			# Clean the image of small objects

			# Define the structuring element for morphological operations
			kernel = np.ones((3,3), np.uint8)
			kernel_opening = cv.getStructuringElement(cv.MORPH_RECT, (7,7))

			# Perform dilation followed by erosion
			iterations = 2
			dilated_image = cv.dilate(threshGaussian, kernel, iterations=iterations)
			image = cv.erode(dilated_image, kernel, iterations=iterations)

			####################################################
			# Apply hough transform

			tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

			hspace, theta, dist = hough_line(image, tested_angles)

			h, q, d = hough_line_peaks(hspace, theta, dist)

			# Any image that has a crazy amount of lines found then there
			# is no actualy line but just artefacts 
			# I found that 4 is good enough, but 10 could be slightly better
			if d.size < 4:
				print(filename)

