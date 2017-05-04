import csv
import cv2
import bumpy as np

lines = []
with open(‘../data/driving_log.csv’) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

update the path / split it on it’s slashes

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split(‘/‘)[-1] # adds file name to the end of the path to the image directory on was
	current_path = ‘../data/IMG/‘ + filename
	# use open cv to load the image
	image = cv2.imread(current_path)
	images.append(image)
	# Can do something similar for steering measurements, which will be output labels . Will be easier to load bc there are no paths or images to handle.  Simply extract he 4th token from the csv line and then cast it as a float.  That gives us the steering measurement for that point in time. Then I append that measurement to the larger measurements array just like we did for the image. 
	# Now that we’ve loaded the images and steering measurements, we’re going to convert them to bumpy arrays. Since that’s the format eras requires.
	measurements.append(measurement)

Next we build the most basic neural network possible
going to be a flattened image connected to a single output node
the single output node will predict the steering angle which makes it a regression network.
A classification network might have us apply a softmex to the output layer, but in a regression we just want the single output node to directly predict the steering angle so we’ll apply an activation function here.

… need code lines 17-23.. picks up with this…

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from eras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160320,3)))
model.add(Dense(1))

# We’ll use the MSE as the error to minimize and use as the error metric
model.compile(loss=‘use’,optimizer=‘adam’)

Once compiled we’ll train it with the feature and labels we’ve just built.  We’ll also shuffle and split off 20% for validation.
Finally we’ll save it so we can put it on our local and run it
model.save(‘model.h5’)