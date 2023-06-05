# Olivier Paredes
# Practical Astronomy Crew
# 23/05/2023
####################################################
#imports
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
import numpy as np 
import cv2

####################################################
# Read the FITS image using astropy
plt.style.use(astropy_mpl_style)
image_file = get_pkg_data_filename('fits_test_1.fit')
fits.info(image_file)
image_data = fits.getdata(image_file, ext=0)
print(image_data.shape)
plt.figure(figsize = (10, 10))
plt.imshow(image_data, cmap='gray', norm=LogNorm())
plt.colorbar()
plt.show()

####################################################
# Hough Transform

image = cv2.imread('test_3.png', 0)

#invert the image
#image = ~image
plt.imshow(image, cmap='gray')

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

hspace, theta, dist = hough_line(image, tested_angles)


plt.figure(figsize = (10, 10))
plt.imshow(hspace)
plt.show()

#Now, to find the location of peaks in the hough space we can use hough_line_peaks
h, q, d = hough_line_peaks(hspace, theta, dist)
print(d.size)

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
def zscale_range(image_data, contrast=0.25, num_points=600, num_per_row=120):

    """
    Computes the range of pixel values to use when adjusting the contrast
    of FITs images using the zscale algorithm.  The zscale algorithm
    originates in Iraf.  More information about it can be found in the help
    section for DISPLAY in Iraf.

    Briefly, the zscale algorithm uses an evenly distributed subsample of the
    input image instead of a full histogram.  The subsample is sorted by
    intensity and then fitted with an iterative least squares fit algorithm.
    The endpoints of this fit give the range of pixel values to use when
    adjusting the contrast.

    Input:  image_data  -- the array of data contained in the FITs image
                           (must have 2 dimensions)
            contrast    -- the contrast parameter for the zscale algorithm
            num_points  -- the number of points to use when sampling the
                           image data
            num_per_row -- number of points per row when sampling
    
    Return: 1.) The minimum pixel value to use when adjusting contrast
            2.) The maximum pixel value to use when adjusting contrast
    """

    # check input shape
    if len(image_data.shape) != 2:
        raise ValueError("input data is not an image")

    # check contrast
    if contrast <= 0.0:
        contrast = 1.0

    # check number of points to use is sane
    if num_points > numpy.size(image_data) or num_points < 0:
        num_points = 0.5 * numpy.size(image_data)

    # determine the number of points in each column
    num_per_col = int(float(num_points) / float(num_per_row) + 0.5)

    # integers that determine how to sample the control points
    xsize, ysize = image_data.shape
    row_skip = float(xsize - 1) / float(num_per_row - 1)
    col_skip = float(ysize - 1) / float(num_per_col - 1)

    # create a regular subsampled grid which includes the corners and edges,
    # indexing from 0 to xsize - 1, ysize - 1
    data = []
   
    for i in xrange(num_per_row):
        x = int(i * row_skip + 0.5)
        for j in xrange(num_per_col):
            y = int(j * col_skip + 0.5)
            data.append(image_data[x, y])

    # actual number of points selected
    num_pixels = len(data)

    # sort the data by intensity
    data.sort()

    # check for a flat distribution of pixels
    data_min = min(data)
    data_max = max(data)
    center_pixel = (num_pixels + 1) / 2
    
    if data_min == data_max:
        return data_min, data_max

    # compute the median
    if num_pixels % 2 == 0:
        median = data[center_pixel - 1]
    else:
        median = 0.5 * (data[center_pixel - 1] + data[center_pixel])

    # compute an iterative fit to intensity
    pixel_indeces = map(float, xrange(num_pixels))
    points = pointarray.PointArray(pixel_indeces, data, min_err=1.0e-4)
    fit = points.sigmaIterate()

    num_allowed = 0
    for pt in points.allowedPoints():
        num_allowed += 1

    if num_allowed < int(num_pixels / 2.0):
        return data_min, data_max

    # compute the limits
    z1 = median - (center_pixel - 1) * (fit.slope / contrast)
    z2 = median + (num_pixels - center_pixel) * (fit.slope / contrast)

    if z1 > data_min:
        zmin = z1
    else:
        zmin = data_min

    if z2 < data_max:
        zmax = z2
    else:
        zmax = data_max

    # last ditch sanity check
    if zmin >= zmax:
        zmin = data_min
        zmax = data_max

    return zmin, zmax