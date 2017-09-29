import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pickle
import cv2
from skimage.feature import hog


data_dir = os.path.join(os.path.curdir, 'data')
csv_file = os.path.join(data_dir, 'object-detection-crowdai', 'labels.csv')
nv_dirs = [os.path.join(data_dir, 'non-vehicles', 'Extras'),
           os.path.join(data_dir, 'non-vehicles', 'GTI')]
v_dirs = [os.path.join(data_dir, 'vehicles', 'GTI_Far'),
          os.path.join(data_dir, 'vehicles', 'GTI_Left'),
          os.path.join(data_dir, 'vehicles', 'GTI_MiddleClose'),
          os.path.join(data_dir, 'vehicles', 'GTI_Right'),
          os.path.join(data_dir, 'vehicles', 'KITTI_extracted')]


def load_data():
    with open(csv_file, newline='') as cf:
        reader = csv.reader(cf, delimiter=',')
        for row in reader:
            print(row)
            # TODO: Parse the data


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


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features