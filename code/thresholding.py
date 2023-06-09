# Olivier Paredes
# Practical Astronomy Crew
###################################################

import cv2 as cv
import numpy as np

###################################################

img = cv.imread('../data/faint_images/faint_3.png')

# Convert the image to greyscale
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image slightly, for faint images
# img = cv.GaussianBlur(img, (7, 7), 0)
# Inverse the image 
img = ~img

_, th1 = cv.threshold(img, 50, 250, cv.THRESH_OTSU)
# (T, threshInv) = cv.threshold(img, 0, 155, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
threshGaussian = cv.adaptiveThreshold(img, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 10)
cv.imshow("image", img)
cv.imshow("th1", th1)
cv.imshow("threshGaussian", threshGaussian)

cv.waitKey(0)
cv.destroyAllWindows()