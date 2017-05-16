**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./figures/Car_Not_a_Car.jpg
[image2]: ./figures/L_channel_L_channel_hog.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./figures/Car_Positions_Heat_Map.jpg
[image5]: ./figures/sliding_window.jpg
[image6]: ./figures/Test_image_Windows.jpg
[image7]: ./examples/output_bboxes.png
[image8]: ./figures/Car_Spatial_binning.jpg
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used multiple functions to extract HOG features in the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different binned color features using `bin_spatial()`, as can be seen in this example:

![alt text][image8]

I then explored different color spaces using the `get_hog()` function and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and the `L` channel of the HLS space:

![alt text][image2]

I then get the feature vector by using the `bin_spatial()` function to compute binned color features, then computing the color histogram features and concatenating them into asingle feature vector. This code can be found in `get_feature_vector()`

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on an optimization of speed and accuracy.   It is the one suggested by many in the forums and slack channels, and I kept coming back to it because it continued to be the best.  I might try to grid search parameters later though, on a higher powered computer / more time, as I think that is the best way to find optimal parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I ended up training an SVM with a Radial Basis Function kernel instead of a Linear SVM because it performed better and gave me less false positives.  The data used was scaled, shuffled, and split as per usual.  Scaled using the `StandardScaler`, shuffled and split using `train_test_split`.  Ultimately, I'd like to use a neural network instead of classical machine learning methods for this project though as I think it'd generalize and estimate better than the approach here.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

Just kidding, I used the suggested sliding windows set up and chose the window sizes to be [(64, 64), (96, 96), (128, 128), (160, 160), (192, 192)].

![alt text][image6]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used the suggested implementation which uses thresholded heat maps and labels.  This approach filters false negatives as well as leaves us with one window for every vehicle instead of all windows around the vehicle.

![alt text][image5]

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_projecced.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


If you take a look at the `process_image()` function, you'll see that the last 5 window frames are saved.  Then you'll see a thresholded heat map applied to those frames, along with labeling.  The resulting image is just the original with the windows from the rest of the process drawn on. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had issues with false positives a great deal of the time.  Way too many non vehicles were being detected as vehicles, and that in the real world leads to trouble.  In the project video in particular, the billboard sign was detected as a vehicle, as were random spots along the highway

As mentioned previously, I think using a neural network trained with the right data and built with the right architecture could result in a better model.  It's uncertain if there will be a tradeoff in speed, but a lot of the feature engineering in this project took very long.  

In addition to neural nets as a solution to improve robustness, we could potentially remove some false positives by combining the lane finding techniques with these.

As it is now, it's not useful for real time processing.
