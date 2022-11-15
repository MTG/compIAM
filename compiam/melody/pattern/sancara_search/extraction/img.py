import numpy as np

import cv2

from scipy import signal
from scipy import misc
from scipy.ndimage import binary_opening
import skimage
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm

scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

sobel = np.asarray([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float)

sobel_y = np.array([[1,  2, 1], 
                    [0,  0, 0], 
                    [-1,-2,-1]], dtype = np.float)

def remove_diagonal(X):
    X_ = X.copy()
    n = X.shape[0]

    for i in range(-30, 30):
        x = range(n)
        y = [x_ + i for x_ in x]
        
        if i != 0:
            x = x[abs(i):-abs(i)]
            y = y[abs(i):-abs(i)]
        X_[x,y] = 0
    return X_


def convolve_array(X, cfilter=scharr):
    grad = signal.convolve2d(X, cfilter, boundary='symm', mode='same')
    X_conv = np.absolute(grad)
    return X_conv

    
def convolve_array_sobel(X, cfilter=sobel):
    grad_x = signal.convolve2d(X, sobel_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(X, sobel_y, boundary='symm', mode='same')
    grad = np.hypot(grad_x, grad_y)
    X_conv = np.absolute(grad)
    return X_conv


def convolve_array_tile(X, cfilter=sobel, divisor=49):
    """
    Iteratively convolve equal sized tiles in X, rejoining for fast convolution of the whole
    """
    x_height, x_width = X.shape

    assert x_height == x_width, "convolve_array expects square matrix"

    # Find even split for array
    divisor = divisor
    tile_height = None
    while (not tile_height) or (int(tile_height) != tile_height):
        # iterate divisor until whole number is found
        divisor += 1
        tile_height = x_height / divisor

    tile_height = int(tile_height)

    # Get list of tiles 
    tiled_array = X.reshape(divisor, tile_height, -1, tile_height)\
                   .swapaxes(1, 2)\
                   .reshape(-1, tile_height, tile_height)

    # Convolve tiles iteratively
    tiled_array_conv = np.array([convolve_array(x, cfilter=cfilter) for x in tiled_array])

    # Reconstruct original array using convolved tiles
    X_conv = tiled_array_conv.reshape(divisor, divisor, tile_height, tile_height)\
                             .swapaxes(1, 2)\
                             .reshape(x_height, x_width)

    return X_conv


def binarize(X, bin_thresh, filename=None):
    X_bin = X.copy()
    X_bin[X_bin < bin_thresh] = 0
    X_bin[X_bin >= bin_thresh] = 1

    if filename:
        skimage.io.imsave(filename, X_bin)

    return X_bin


def diagonal_gaussian(X, gauss_sigma, filename=False):
    d = X.shape[0]
    X_gauss = X.copy()

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        diy = np.append(diag_indices_y, diag_indices_y[:i])
        diy = diy[i:]
        X_gauss[diag_indices_x, diy] = gaussian_filter(X_gauss[diag_indices_x, diy], sigma=gauss_sigma)

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        dix = np.append(diag_indices_x, diag_indices_x[:i])
        dix = dix[i:]
        X_gauss[dix, diag_indices_y] = gaussian_filter(X_gauss[dix, diag_indices_y], sigma=gauss_sigma)

    if filename:
        skimage.io.imsave(filename, X_gauss)

    return X_gauss


def make_symmetric(X):
    return X + X.T - np.diag(X.diagonal())


def edges_to_contours(X, kernel_size=10):
    X_copy = X.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    close = cv2.morphologyEx(X_copy, cv2.MORPH_CLOSE, kernel)
    X_copy = close-X_copy
    X_copy[X_copy==-1]=0
    return close#X_copy


def apply_bin_op(X, binop_dim):
    binop_struct = np.zeros((binop_dim, binop_dim))
    np.fill_diagonal(binop_struct, 1)
    X_binop = binary_opening(X, structure=binop_struct).astype(np.int)

    return X_binop


def plot_hough(image, h, theta, d, peaks, out_file):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()


def hough_transform(X, min_dist_sec, cqt_window, hough_high_angle, hough_low_angle, hough_threshold, filename=None):
    # TODO: fix this
    hough_min_dist = int(min_dist_sec * cqt_window)

    if hough_high_angle == hough_low_angle:
        tested_angles = np.array([-hough_high_angle * np.pi / 180])
    else:
        tested_angles = np.linspace(- hough_low_angle * np.pi / 180, -hough_high_angle-1 * np.pi / 180, 100, endpoint=False) #np.array([-x*np.pi/180 for x in range(43,47)])

    h, theta, d = hough_line(X, theta=tested_angles)
    peaks = hough_line_peaks(h, theta, d, min_distance=hough_min_dist, min_angle=0, threshold=hough_threshold)
    
    if filename:
        plot_hough(X, h, theta, d, peaks, filename)

    return peaks


def hough_transform_new(X, hough_high_angle, hough_low_angle, hough_threshold, filename=None):

    lines = cv2.HoughLines(X.astype(np.uint8),1, np.pi/180, hough_threshold)
    lines = lines[:,0]
    
    upper_angle = ((90 + hough_high_angle) * np.pi / 180)
    lower_angle = ((90 + hough_low_angle) * np.pi / 180)
    
    lines = np.array([[dist, angle] for dist, angle in lines if lower_angle < angle < upper_angle])
    
    peaks = (list(range(len(lines[:,1]))), lines[:,1], lines[:,0])
    
    if filename:
        plot_hough_new(X, peaks, filename)

    return peaks


def plot_hough_new(X, peaks, out_file):
    fig, ax = plt.subplots(figsize=(30, 12))

    ax.imshow(X, cmap=cm.gray)
    ax.set_ylim((X.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Detected lines')

    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='red', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()



#   filter_size = 5

#   h, w = X_cont.shape

#   X_fill = np.zeros((h, w))

#   def get_centered_array(X, x, y, s):
#       """
#       Return <s>x<s> array centered on <x> and <y> in <X>
#       Any part of returned array that exists outside of <X>
#       is be filled with nans
#       """
#       o, r = np.divmod(s, 2)
#       l = (x-(o+r-1)).clip(0)
#       u = (y-(o+r-1)).clip(0)
#       X_ = X[l: x+o+1, u:y+o+1]
#       out = np.full((s, s), np.nan, dtype=X.dtype)
#       out[:X_.shape[0], :X_.shape[1]] = X_
#       return out


#   def is_surrounded(X):
#       """
#       Is the center square in x sufficiently surrounded by 
#       non zero 
#       """
#       triu = np.triu(X)
#       tril = np.tril(X)
#       return 1 in triu and 1 in tril

#   import tqdm
#   for i in tqdm.tqdm(range(x_height)):
#       for j in range(x_width):
#           if i > j:
#               continue
#           cent_X = get_centered_array(X_cont, i, j, filter_size)
#           if is_surrounded(cent_X):
#               X_fill[i,j] = 1

#   X_fill = X_fill + X_fill.T - np.diag(np.diag(X_fill))

#   if save_imgs:
#       skimage.io.imsave(merg_filename, X_fill)








