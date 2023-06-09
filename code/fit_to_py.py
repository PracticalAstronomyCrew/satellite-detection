# Olivier Paredes
# Practical Astronomy Crew
####################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits

# Open the FITS file
hdul = fits.open('../data/faint_images/230518_Li_.00000083.Mouse_click_position.FIT')

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
threshGaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 50)
cv.imshow("image", img)
cv.imshow("threshGaussian", threshGaussian)
cv.waitKey(0)