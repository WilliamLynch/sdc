## Writeup - Advanced Lane Finding Project
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/distortion/3_undistorted_img.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image9]: ./examples/distortion/straight_lines1_undistorted_img.jpg "Undistorted and Orig"
[image3]: ./examples/thresholding/ThresholdingCombined.jpg "Thresholding Combined"
[image4]: ./examples/transformation/transformcombination.jpg "Warp Example"
[image5]: ./examples/detection/windowfittingresults.jpg "Window Fitting"
[image7]: ./examples/detection/slidingwindowpolyfit.jpg "Fit Visual"
[image6]: ./examples/detection/curvature.jpg "Output"
[image8]: ./examples/detection/histogram.jpg "Histogram"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "project_notebook.ipynb",

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  For this part I used cv2.drawChessboardCorners() to automatically detect and draw the corners.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function ...

```python
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

and obtained this result:

![alt text][image1]

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied `cal_undistort()` (in the Correcting for Distortion section of `project_workbook.ipynb`) to some test images to make sure it'd work on new images. 

![alt text][image9]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Then, I used a combination of color and gradient thresholds to generate a binary image (thresholding steps found in `pipeline()` function of `project_workbook.ipynb` as well as separately in the Thresholding Section).  In particular, we convert our image to HSV space and separate the V channel.  We then apply each of our thresholding functions (Sobel Operator function, Magnitude of the Gradient function, and Direction of the Gradient function) on our images so that we can optimally detect which pixels are likely to be part of lane lines.  Next in the pipeline is color thresholding.  

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in my workbook `project_workbook.ipynb` (in the Transforms of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that I have a calibrated, thresholded, and transformed image, I can look at the peaks in a histogram to identify lane-line pixels.

![alt text][image8]

Once we have this histogram, we can see the two largest peaks and use that as a starting point for our sliding window search.  As you'll see in my workbook `project_workbook.ipynb` (in the Detect Lane Lines section of the IPython notebook), we choose our number of sliding windows, along with the height and width of said windows, the minimum number of pixels we need to recenter the image after each slide, and two lists to append our left and right lane indices.

Then, for each window, we find the window boundaries, draw windows onto our image, Identify the nonzero pixels in x and y within the window, append those indices to our lists, recenter the window, and repeat.

Once that process is finished, we take the pixel positions we've been collecting in our lists, and fit a second order polynomial to each (left lane x, left lane y, right lane x, right lane y)

![alt text][image7]
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my workbook `project_workbook.ipynb` (in the Lane Line Curve code section of the IPython notebook)

Essentially, since we have a thresholded image, a list of pixels which we estimate belong to lane lines, and we've fit a polynomial to those pixel positions, we can measure the radius of curvature of our fit.

To do this we define y value where we want to determine the radius of curvature and for this project it'll be a pixel at the bottom of the image (max y value).
It looks like this

```python
ym_per_pix = 30/720 
xm_per_pix = 3.7/700
y_eval = 700

y1 = (left_fit[0]*y_eval*2 + left_fit[1])*xm_per_pix/ym_per_pix
y2 = left_fit[0]*xm_per_pix*2/(ym_per_pix*ym_per_pix)

curvature = ((1 + (y1**2))**(1.5))/np.absolute(y2)
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this in my workbook `project_workbook.ipynb` (in the Determine the lane curvature code section of the IPython notebook)

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline will likely fail in the transform section as the source and destination points are very sentitive to change.  Small changes affected how well lane lines were deteted in my binary warped image greatly.

The video makes it work well, but I doubt it'd work out in the road in it's current state.

I definitely think my pipeline can be improved, as it fails when I run it on the harder video. 