import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten,Lambda, Cropping2D,MaxPooling2D,Dropout,Activation, Dense, ELU
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import ast
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from preprocessor import *

# Read data file in array
logs = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        logs.append(line)

logscv = logs

# Generator of train data
def image_generator(logs, batch_size=128):
    sample_size = len(logs)
    while True:          
        batch_X = []
        batch_y = []
        for i in range(batch_size):  
            try:
                # Random selection of image            
                idx = np.random.randint(sample_size)   
                data = [i.split('\t', 1)[0] for i in logs[idx]]
                data = [i.strip() for i in data]
                selection = np.random.choice(['center', 'left', 'right'])

                # Random selection of image            
                # For right and left image add .25 to steering angle  
                # This is for correction: advice from the slack said to try .25
                # and then .20 / .30 / .35 / etc until improvement
                if selection == 'center':
                    source_path = data[0]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    image = cv2.imread(current_path)

                    #Steering
                    source_path = data[3]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    angle=ast.literal_eval(data[3])  
                    #angle = float(data[3])
                    image_process, anglenew= preprocess_image(image, float(angle))

                if selection == 'left':    
                    source_path = data[1]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    image = cv2.imread(current_path)

                    # Steering
                    source_path = data[3]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    angle=ast.literal_eval(data[3])
                    #angle = float(data[3])
                    image_process, anglenew= preprocess_image(image, (float(angle)+.25))

                if selection == 'right':      
                    source_path = data[2]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    image = cv2.imread(current_path)

                    # Steering
                    source_path = data[3]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    current_path = './data/' + filename
                    angle=ast.literal_eval(data[3])
                    #angle = float(data[3])
                    
                    image_process, anglenew= preprocess_image(image, (float(angle)-.25))

                # Remove steering angle '0' randomly to improve our CNN 
                flip_prob = np.random.uniform()
                if (flip_prob >= .5 and (anglenew >-.1 and anglenew <.1)):            
                    batch_X.append(image_process)
                    batch_y.append(anglenew)          
                else: 
                    batch_X.append(image_process)
                    batch_y.append(anglenew)
            except:
                pass
            
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)        
        batch_X, batch_y= shuffle(batch_X, batch_y)
        
        yield (batch_X, batch_y)        

# Generator of validation data
def image_val_generator(logscv, batch_size=128):         
    while True:          
        batch_X = []
        batch_y = []    
        for i in range(batch_size):  
            try:

                # Randomly select image (only center data)
                idx = np.random.randint(len(logscv))                            

                data = [i.split('\t', 1)[0] for i in logscv[idx]]
                data = [i.strip() for i in data]

                # Prepare image
                source_path = data[0]
                filename = source_path.split('/')[-1]
                filename = source_path.split('\\')[-1]
                current_path = './data/' + filename
                image = cv2.imread(current_path)
                image = image[30:140,50:270]
                image = cv2.resize(image, (200,66))
                image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                
                # Steering
                source_path = data[3]
                filename = source_path.split('/')[-1]
                filename = source_path.split('\\')[-1]
                current_path = './data/' + filename

                angle= ast.literal_eval(data[3])  
                #angle = float(data[3]) - was acting funny so I changed it

                # Flip data
                flip_prob = np.random.uniform()
                if flip_prob > .5:
                    image=cv2.flip(image,1)
                    if angle != 0:
                        angle = -angle
            except:
                pass                    
                    
            batch_X.append(image)
            batch_y.append(angle)      
                        
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        batch_X, batch_y= shuffle(batch_X, batch_y)

        yield (batch_X, batch_y)  

def nvidia():
    # Implement the Nvidia CNN
    model = Sequential()
    
    # In this first layer, I tried to normalize the data and augment the images
    # This is supposed to help remove unnecessary background noise 
    # so we can see the road better.  Didn't work so I removed it
    #model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160,320,3)))
    
    # Crop
    # Ended up removing this as well as it didn't work as designed
    #model.add(Cropping2D(cropping=((70,25), (0,0))))
    
    # In this layer we perform feature extraction.
    # The values we're using come from the NVIDIA architecture and have been
    # tested thoroughly.
    # We use 5 convolutional layers
    # The first 3 use a 5x5 kernel with a 2x2 stride, elu activation,
    # and a dropout layer
    model.add(Conv2D(24, 5, 5, subsample=(2,2), input_shape=(66,200,3,),  border_mode='valid'))
    # Elu
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    model.add(Conv2D(36, 5, 5, subsample=(2,2),  border_mode='valid'))
    # Elu
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    model.add(Conv2D(48, 5, 5, subsample=(2,2),  border_mode='valid'))
    # Elu
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    
    # The last 2 use a 3x3 kernel with a 1x1 stride, elu activation,
    # and a dropout layer
    model.add(Conv2D(64, 3, 3,  border_mode='valid'))
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 3, 3,  border_mode='valid'))
    model.add(ELU())
    
    # Finally, we end with three fully connected layers which controls steering.
    model.add(Flatten())
    # Fully Connected
    model.add(Dense(100))
    # Elu
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    # Fully Connected
    model.add(Dense(50))
    # Elu
    model.add(ELU())
    # Dropout technique to prevent overfitting.
    model.add(Dropout(0.3))
    # Fully Connected
    model.add(Dense(10))
    model.add(ELU())
    # Output
    model.add(Dense(1))

    return model

# commaai model
def commaai():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160,320,3)))
    #Crop
    model.add(Cropping2D(cropping=((65,25), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model

model = nvidea()

# This check_point saves the best version of our model
# based on cv mse error
check_point = ModelCheckpoint('model.h5',
                              monitor='val_loss',
                              mode='min',
                              verbose=1,
                              save_best_only=True)
    
# Early Stopping is a great feature which 
# you guessed it, stops training when there is no improvment. 
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=3,
                              verbose=1)

# Compilation of model with Adam optimizer
# We’ll use the MSE as the error to minimize and use as the error metric
model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
  
# Now that we've trained and tested our model we’ll save it 
# so we can put it on our local and run it.
model.save('model.h5')
print('model saved')

# Runs model
history = model.fit_generator(image_generator(logs),
                              samples_per_epoch=190*128, 
                              nb_epoch=10, 
                              verbose=1,
                              callbacks=[check_point,earlystopping],
                              validation_data=image_val_generator(logscv),
                              nb_val_samples=50*128
                             )

print(history.history.keys())

with plt.style.context('fivethirtyeight'): 
    plt.figure(figsize=(20,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Nvidia Architecture Model - Mean Squared Error Loss')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Error', 'Validation Error'], loc='upper right')
    plt.savefig('./examples/mse_loss.png')