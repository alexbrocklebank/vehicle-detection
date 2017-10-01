import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
import pickle
import cv2
import PIL
from PIL import Image
from skimage.feature import hog
from scipy.ndimage.measurements import label

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


def im_a_pickle_morty(path):
    if os.path.isfile(path):
        with open(path, "rb"):
            try:
                return True
            except StandardError:
                return False
    else:
        return False


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


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
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32) / 255
    bboxes = []

    for i in range(0, len(ystart)):
        img_tosearch = img[ystart[i]:ystop[i], :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        # TODO: Test the color conversion above
        if scale[i] != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (
                np.int(imshape[1] / scale[i]), np.int(imshape[0] / scale[i])))

        #print("Image to Search Dimensions: {} at Scale {}.".format(
        #    ctrans_tosearch.shape, scale[i]))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        #print("Image to Search Dimensions: {} at Scale {}.".format(
        #    ctrans_tosearch.shape, scale[i]))

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        #print("Blocks/win: {}, Cells/step: {}, X/Y Steps: {}/{}".format(
        #    nblocks_per_window, cells_per_step, nxsteps, nysteps
        #))

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))


                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                    (window, window))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale[i])
                    ytop_draw = np.int(ytop * scale[i])
                    win_draw = np.int(window * scale[i])
                    bboxes.append(((xbox_left, ytop_draw + ystart[i]),
                                   (xbox_left + win_draw,
                                    ytop_draw + win_draw + ystart[i])))

    return draw_img, bboxes


def add_heat(heatmap, bbox_list):
    #set minimum pixel width of bounding boxes that could contain cars.
    m = 15
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if (abs(box[0][1] - box[1][1]) > m) and (abs(box[0][0] - box[1][0]) > m):
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def equalize(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] = np.amax(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]])
    return heatmap


def average_heatmap(heatmaps):
    n = len(heatmaps)
    average = np.zeros(heatmaps[0].shape, np.float)

    for map in heatmaps:
        average = average + map / n

    average = np.array(np.round(average), dtype=np.uint8)
    return average


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img
