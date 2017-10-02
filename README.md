**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle.png
[image2]: ./output_images/nonvehicle.png
[image3]: ./output_images/test1.png
[image4]: ./output_images/test2.png
[image5]: ./output_images/test3.png
[image6]: ./output_images/vid_frame.png
[video1]: ./output_video/output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is referenced on lines 85 and 102 of `vehicle-track.py` for Cars and Noncars respectively.  These lines call the function `extract_features()` which is contained in `assistant_functions.py` on lines 99 to 164.  This function calls and combines the features produced from `bin_spatial()` (lines 81-85), `color_hist()` (lines 88-96), and `get_hog_features()` (lines 60-78).

I started by reading in all the `vehicle` and `non-vehicle` images from the `data` subfolder.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle Image][image1]
![Non-Vehicle Image][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and explored different color spaces on the test images provided.  I made various tweaks by passing in different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) and noted the amount of boxes on the output.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the LinearSVC class from sklearn.svm.  The classifier is loaded and stored using the `joblib` library, to save me time retraining the classifier and allow easy reuse.  If the parameters are changed, or if the file does not exist, it creates and trains the classifier.  The classifier is created on line 147 in `vehicle-track.py` and fit on line 157.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to recycle the `find_cars()` method from the lesson since it performed the fastest and was very robust.  I modified the `ystart`, `ystop`, and `scale` variables in the call of the function to become tuples.  This way I can iterate through different "size" windows in different areas of the source image and at various scales.

I actually determined the window scale sizes first by multiples of the 32x32 training images, but then I realized this tactic made finding the cars hit-or-miss entirely, with very few bounding boxes resulting.  Instead I reduced the scale, and more carefully set the y-ranges.  This yielded many more bounding boxes which I could set thresholds on.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap with a minimum bounding box width/height to prevent small rectangles from appearing due to the averaging I apply later.  The heatmap is then thresholded to identify most likely vehicle positions.  I add this heatmap to a `heatmaps` deque that I then average over the last 10 frames to produce a new heatmap with noise of false-positives reduced.  Then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here the resulting bounding boxes are drawn onto the video frame:
![Annotated Video Frame][image6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline seems to find false positives in dark, shadowy environments still.  It also seems very bad at detecting white cars no matter how much I tweaked the parameters.  I will have to do much more further research on color spaces and possibly add a grayscale HOG channel for a total of 4 channels.  I would also like to try to train a neural network to perform this task when I have the time, I feel like this would be much easier and faster with a CNN.

#### 2. Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I essentially started with the data processing of manipulating the images and their labels, and then moved into storing the data into pickle files to be able to save as much time as possible.  I then moved step by step on the pipeline, in which the find_cars() and extract_features() functions from the lesson were immensely helpful in getting me started.  I processed the test images to perfect the parameters to prevent all the time needed to process the video.  Once I had good performance on images I was able to add in the video code, then I was able to add heatmap averaging and methods to reduce false positives in time sequences.

I am actually planning on continuing this project for some time, I want to try and weigh performance between various SVMs and Convolutional Neural Networks as well.  The main failure I notice is when cars are seen more from the side than the rear (since rear images are most of the dataset) or at further distances, where I can't shrink the scale of the detection because small pieces of the image give many false positives.  I plan on taking the extra crowdai dataset and process the annotated images into equal vehicle and non-vehicle sets.  From this, as well as some video I've been taking with my dashcam, I should be able to have more than enough data to train robust versions of both classifier types.



