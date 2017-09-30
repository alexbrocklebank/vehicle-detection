import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from skimage.feature import hog
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
from assistant_functions import *

# Read in Vehicle and Not-Vehicle images
cars, notcars = load_data()

# TODO: Sliding Window Function

# Classifier Parameters
classifier_pickle = 'classifier.pkl'
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [450, None]  # Min and max in y to search in slide_window()
write_params = False

# Attempt to load previously stored parameters
parameters = {}
try:
    loaded_params = pickle.load(open('parameters.pkl', 'rb'))
    print("Parameters loaded from file.")
except (OSError, IOError) as e:
    write_params = True

# Store parameters in dictionary
param_names = ('classifier_pickle', 'color_space', 'orient', 'pix_per_cell',
               'cell_per_block', 'hog_channel', 'spatial_size', 'hist_bins',
               'spatial_feat', 'hist_feat', 'hog_feat', 'y_start_stop')
for i in param_names:
    if not write_params:
        if loaded_params[i] != locals()[i]:
            write_params = True
    parameters[i] = locals()[i]

# Store dictionary of parameters to pickle
if write_params:
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(parameters, f)
        print("Parameters written to file.")

# Obtain Features
if (not write_params) and im_a_pickle_morty("car_features.pkl"):
    car_features = joblib.load("car_features.pkl")
    print("Car Features loaded from file.")
else:
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    joblib.dump(car_features, "car_features.pkl")
    print("Car Features written to file.")

if (not write_params) and im_a_pickle_morty("notcar_features.pkl"):
    notcar_features = joblib.load("notcar_features.pkl")
    print("Notcar Features loaded from file.")
else:
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    joblib.dump(notcar_features, "notcar_features.pkl")
    print("Notcar Features written to file.")

# Normalize/Scale Features
scaled_X = []
y = []
if not write_params:
    if im_a_pickle_morty("X_scaled.pkl"):
        scaled_X = joblib.load("X_scaled.pkl")
        print("Scaled X loaded from file.")
    if im_a_pickle_morty("y_labels.pkl"):
        y = joblib.load("y_labels.pkl")
        print("Y Labels loaded from file.")
if (len(scaled_X) == 0) or (len(y) == 0):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    joblib.dump(scaled_X, "X_scaled.pkl")
    print("Scaled X written to file.")
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    joblib.dump(y, "y_labels.pkl")
    print("Y Labels written to file.")
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Create Classifier
svc = LinearSVC()

# Train Clasifier if it hasn't already been stored & parameters are unchanged
if (not write_params) and im_a_pickle_morty(classifier_pickle):
    svc = joblib.load(classifier_pickle)
    print("Classifier loaded...")
else:
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    # Pickle the classifier for re-use
    joblib.dump(svc, classifier_pickle)
    print("Classifier saved.")

# TODO: Receive Frame

# TODO: Find Vehicles

# TODO: Heatmap
# TODO: Centroid of Duplicates

# TODO: Record positions of found vehicles
