import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from assistant_functions import *


# TODO: Declare Variables

# TODO: Receive Input Data and Labels

# TODO: Color Features
# TODO: Gradient Features

# TODO: Normalize/Scale Features

# TODO: Sliding Window Function

# TODO: Classifier Parameters
# TODO: Create Classifier

# TODO: Train Clasifier

# TODO: Receive Frame

# TODO: Find Vehicles

# TODO: Heatmap
# TODO: Centroid of Duplicates

# TODO: Record positions of found vehicles