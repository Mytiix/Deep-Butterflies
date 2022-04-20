# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:18:08 2022

@author: Navdeep Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import glob
import math
import sys
import albumentations as A
import tensorflow as tf
import cv2 as cv
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import utils
from landmark_HM_models import *
from datetime import datetime
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

#========================== Functions =========================================
def parse_data(image, lm):
    """
    Reads image and landmark files depending on
    specified extension.
    """
    image_content = tf.io.read_file(image)

    chan = channels[0] if color == 'rgb' else channels[1]
    image = tf.io.decode_png(image_content, channels=chan)
    image = tf.image.resize(image, (256,256))
    
    #image = tf.image.resize_with_pad(image, 256,256) #if original images to be used to parse the data
    
    return  image, lm

def normalize(image, lm):
    image = tf.cast(image, tf.float32)/255.0
    return image, lm


transforms = A.Compose([
                       A.RandomBrightnessContrast(),
                       A.Affine(scale=(0.8,1), 
                                translate_px= (0, 10), 
                                rotate= (-10,10), 
                                shear=(0,10), 
                                interpolation=0, #nearest
                                mode=2, fit_output=False), #mode replicate
                       A.HorizontalFlip(p=0.3),
                       ], 
                        keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))



def aug_fn(image, lm):
    image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
    lm = lm.numpy()
    aug_data = transforms(image=image, keypoints = lm)
    image_aug = aug_data['image'] 
    lm_aug = aug_data['keypoints']
    lm_aug = np.array(lm_aug, dtype=np.int32)
    
    return image_aug, lm_aug

def aug_apply(image,lm):
    image, lm = tf.py_function(aug_fn, (image, lm), (tf.float32, tf.float32))
    image.set_shape(image_size)
    lm.set_shape((N,2))
    return image, lm

def _exp(xL, yL, sigma, H, W):  # Exponential kernel for genrating heatmaps
    xx, yy = np.mgrid[0:H:1, 0:W:1]
    kernel = np.exp(-np.log(2) * 0.5* (np.abs(yy - xL) + np.abs(xx - yL)) / sigma)
    kernel = np.float32(kernel)
    
    return kernel

# convert image to heatmap
def _convertToHM(img, keypoints):

        #sigma = 3
        H = img.shape[0]
        W = img.shape[1]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints), dtype=np.float32)

        for i in range(0, nKeypoints):
            x = keypoints[i][0]
            y = keypoints[i][1]

            channel_hm = _exp(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm
        
        img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints, 1))

        return img, img_hm

def to_hm(image, lm):
    image, lm = tf.py_function(_convertToHM, (image,lm), (tf.float32, tf.float32))
    image.set_shape(image_size)
    lm.set_shape((256*256*N, 1))
    return image, lm  

#==============================================================================
if __name__ == "__main__":
    
    """
    1 = training repository (in the form  of'repository/images/' and 'repository/landmark/')
    2 = number of epochs to train the model
    3 = batch_size
    4 = sigma (heatmap spread)
    """

    # Params
    species = 'all'
    side = 'v'
    color = 'rgb'
    
    repository = 'D:/Dataset_TFE/images_v2/'+species+'/'+side+'/training/rescaled'
    filename = species+'_'+side+'_'+color
    n_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    sigma = int(sys.argv[3])
    #model_name = int(sys.argv[5])  #UNET or FCN8
    
    image_size = (256,256,3) if color == 'rgb' else (256,256,1) # input image size to the model
    channels = [3,1]  #for decoding images in tf 3:RGB, 1:gray
    N = 14 if side == 'v' else 18 #number of landmarks
    seed = 44 # to generate the same indexes for each re-run
    
    image_paths = glob.glob(os.path.join(repository,'images/*.png'))
    lm_paths = glob.glob(os.path.join(repository,'landmarks/*.txt'))
    
    idxs = []
    kp_list = []
    for i in range(len(lm_paths)):  # loading landmarks as list of arrays
        idxs.append(i)
                
        kp = np.loadtxt(lm_paths[i], delimiter=' ')
        kp = np.array(kp).astype(np.int32)
        #kp = kp.flatten('C')
        kp_list.append(kp)

#============== tf dataset generation =========================================
        
    train_idxs, val_idxs = train_test_split(idxs, test_size=0.10, random_state=seed) 
    
    train_images = [image_paths[i] for i in train_idxs]
    train_lmks = [kp_list[i] for i in train_idxs]
    val_images = [image_paths[i] for i in val_idxs]
    val_lmks = [kp_list[i] for i in val_idxs] 
    
    
    steps_per_epoch = math.ceil(len(train_images) / batch_size)
    val_steps = math.ceil(len(val_images) / batch_size)
    
#=========== train dataset ====================================================
    tr_ds = tf.data.Dataset.from_tensor_slices((train_images, train_lmks))
    tr_ds = tr_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tr_ds = tr_ds.map(aug_apply, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tr_ds = tr_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tr_ds = tr_ds.map(to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tr_ds = tr_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            random.randint(0, len(image_paths))).batch(batch_size).repeat()
#=========== validation dataset (no augmentation) =============================    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_lmks))
    val_ds = val_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)   
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
            random.randint(0, len(image_paths))).batch(batch_size).repeat()
#================= model building and training ================================    
    model_name = 'unet' # or FCN8
    filepath="./lm_scripts/saved_models/unet/"+model_name+str(1)+'_'+filename+'_sigma'+str(sigma)+".hdf5"
    log_dir="./lm_scripts/logs/fit/"+datetime.now().strftime("%Y%m%d-%H%M%S")
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=50, mode='min', min_lr=0.00001)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                      write_images=False)
    #callbacks = [checkpoint, tensor_board, early_stopping, reduce_lr]
    callbacks = [checkpoint]
        
    optim = RMSprop(learning_rate=0.001)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mse = tf.keras.metrics.MeanSquaredError()
    model = UNET(input_shape=image_size)
    model.compile(loss="mse", optimizer=optim, metrics=[rmse])
    history = model.fit(tr_ds, epochs=n_epochs, callbacks=callbacks,
                            validation_data=val_ds, steps_per_epoch=steps_per_epoch,
                            validation_steps= val_steps,verbose=1)
    

    
    
    
    
    
        
    
        
    
        
    
        
    
        
    
        
    
    

    



