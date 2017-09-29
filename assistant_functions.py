import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
import pickle
import cv2
from skimage.feature import hog


# Data Directories
data_dir = os.path.join(os.path.curdir, 'data')
csv_file = os.path.join(data_dir, 'object-detection-crowdai', 'labels.csv')
nv_dir = os.path.join(data_dir, 'non-vehicles')
v_dir = os.path.join(data_dir, 'vehicles')


def load_csv_data():
    with open(csv_file, newline='') as cf:
        reader = csv.reader(cf, delimiter=',')
        for row in reader:
            print(row)
            # TODO: Parse the data


def load_data():
    notvehicles = glob.glob(nv_dir + os.sep + '**' + os.sep + '*.png',
                            recursive=True)
    vehicles = glob.glob(v_dir + os.sep + '**' + os.sep + '*.png',
                         recursive=True)
    # TODO: Include a check for image size, and sample down if data is too large
    return vehicles, notvehicles


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(
                                  cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features