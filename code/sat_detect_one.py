# Olivier Paredes
# Practical Astronomy Crew
####################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from skimage.morphology import remove_small_objects
from astropy.io import fits
from skimage.transform import hough_line, hough_line_peaks

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

# for faint images chanage the last parameter to lowest value
threshGaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 25)

####################################################

# Define the structuring element for morphological operations
kernel = np.ones((3,3), np.uint8)
kernel_opening = cv.getStructuringElement(cv.MORPH_RECT, (7,7))

# Perform dilation followed by erosion
iterations = 2
dilated_image = cv.dilate(threshGaussian, kernel, iterations=iterations)
image = cv.erode(dilated_image, kernel, iterations=iterations)

####################################################

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

hspace, theta, dist = hough_line(image, tested_angles)

##############################################################

#Example code from skimage documentation to plot the detected lines
angle_list=[]  #Create an empty list to capture all angles

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + hspace),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
             cmap='gray', aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap='gray')

origin = np.array((0, image.shape[1]))

for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
    angle_list.append(angle) #Not for plotting but later calculation of angles
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

#####################################################

