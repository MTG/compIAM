import numpy as np

import cv2

from scipy import signal
from scipy import misc
from scipy.ndimage import binary_opening

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm

scharr = np.array(
    [
        [-3 - 3j, 0 - 10j, +3 - 3j],
        [-10 + 0j, 0 + 0j, +10 + 0j],
        [-3 + 3j, 0 + 10j, +3 + 3j],
    ]
)  # Gx + j*Gy

sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


def remove_diagonal(X):
    X_ = X.copy()
    n = X.shape[0]

    for i in range(-30, 30):
        x = range(n)
        y = [x_ + i for x_ in x]

        if i != 0:
            x = x[abs(i) : -abs(i)]
            y = y[abs(i) : -abs(i)]
        X_[x, y] = 0
    return X_


def convolve_array(X, cfilter=scharr):
    grad = signal.convolve2d(X, cfilter, boundary="symm", mode="same")
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
    tiled_array = (
        X.reshape(divisor, tile_height, -1, tile_height)
        .swapaxes(1, 2)
        .reshape(-1, tile_height, tile_height)
    )

    # Convolve tiles iteratively
    tiled_array_conv = np.array(
        [convolve_array(x, cfilter=cfilter) for x in tiled_array]
    )

    # Reconstruct original array using convolved tiles
    X_conv = (
        tiled_array_conv.reshape(divisor, divisor, tile_height, tile_height)
        .swapaxes(1, 2)
        .reshape(x_height, x_width)
    )

    return X_conv


def binarize(X, bin_thresh):
    X_bin = X.copy()
    X_bin[X_bin < bin_thresh] = 0
    X_bin[X_bin >= bin_thresh] = 1
    return X_bin


def diagonal_gaussian(X, gauss_sigma):
    d = X.shape[0]
    X_gauss = X.copy()

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1, d):
        diy = np.append(diag_indices_y, diag_indices_y[:i])
        diy = diy[i:]
        X_gauss[diag_indices_x, diy] = gaussian_filter(
            X_gauss[diag_indices_x, diy], sigma=gauss_sigma
        )

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1, d):
        dix = np.append(diag_indices_x, diag_indices_x[:i])
        dix = dix[i:]
        X_gauss[dix, diag_indices_y] = gaussian_filter(
            X_gauss[dix, diag_indices_y], sigma=gauss_sigma
        )

    return X_gauss


def make_symmetric(X):
    return np.maximum(X, X.transpose())


def edges_to_contours(X, kernel_size=10):
    X_copy = X.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    close = cv2.morphologyEx(X_copy, cv2.MORPH_CLOSE, kernel)
    X_copy = close - X_copy
    X_copy[X_copy == -1] = 0
    return close  # X_copy


def apply_bin_op(X, binop_dim):
    binop_struct = np.zeros((binop_dim, binop_dim))
    np.fill_diagonal(binop_struct, 1)
    X_binop = binary_opening(X, structure=binop_struct).astype(int)

    return X_binop
