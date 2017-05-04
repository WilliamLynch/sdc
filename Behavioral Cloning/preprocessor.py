import csv, cv2
import numpy as np

# Translate image 
def trans_image(image,steer,tr_range):    
    rows, cols, chans = image.shape

    flip_prob = np.random.uniform()
    # Horizontal translation and 0.008 steering compensation per pixel    
    tr_x = tr_range*flip_prob-tr_range/2
    steer_ang = steer + tr_x/tr_range*.4    
    
    trans_m = np.float32([[1,0,tr_x],[0,1,0]])
    image_tr = cv2.warpAffine(image,trans_m,(cols,rows))
    
    return image_tr,steer_ang

# Crop image 
def crop_image(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]

# Image Processing 
def preprocess_image(image, angle):        
    shape_y = image.shape[0]
    shape_x = image.shape[1]
    
    # Normalize image to HSV 
    image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        
    # Translate image     
    tr_range = 50
    image, angle = trans_image(image, angle, tr_range)
    
    # Crop image   
    image = crop_image(image, 20, 140, 0+tr_range, shape_x-tr_range)
    
    # Resizing
    res = cv2.resize(image,(200,66))             
    
    # Flip image randomly
    angle_cor = angle     
    flip_prob = np.random.uniform()
    if flip_prob > .5:
        res=cv2.flip(res,1)
        if angle_cor!=0:
            angle_cor = -angle            
            
    return res, np.float32(angle_cor)
